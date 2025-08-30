#!/usr/bin/env python3
"""
Simple stress test script for the crowdsourcing backend
"""
import asyncio
import aiohttp
import time
import json
from dataclasses import dataclass
from typing import List, Dict, Any
import argparse

@dataclass
class TestResult:
    success_count: int = 0
    error_count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []

class BackendStressTester:
    def __init__(self, base_url: str, concurrent_users: int = 50):
        self.base_url = base_url.rstrip('/')
        self.concurrent_users = concurrent_users
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10),
            connector=aiohttp.TCPConnector(limit=200, limit_per_host=100)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_get_state(self, session_id: str) -> tuple[bool, float, str]:
        """Test the /api/get-state endpoint"""
        start_time = time.time()
        try:
            async with self.session.get(
                f"{self.base_url}/api/get-state",
                headers={"User-Agent": f"StressTest-{session_id}"}
            ) as response:
                elapsed = time.time() - start_time
                if response.status == 200:
                    data = await response.json()
                    return True, elapsed, ""
                else:
                    return False, elapsed, f"HTTP {response.status}"
        except Exception as e:
            elapsed = time.time() - start_time
            return False, elapsed, str(e)
    
    async def test_submit_response(self, session_id: str) -> tuple[bool, float, str]:
        """Test submitting a response"""
        start_time = time.time()
        try:
            # Realistic response data
            response_data = {
                "joints": {
                    "joint_0": 0.1,
                    "joint_1": -0.2,
                    "joint_2": 0.3,
                    "joint_3": -0.4,
                    "joint_4": 0.5,
                    "joint_5": -0.6,
                    "left_carriage_joint": 0.02
                },
                "gripper_motion": 1
            }
            
            async with self.session.post(
                f"{self.base_url}/api/submit-goal",
                json=response_data,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": f"StressTest-{session_id}"
                }
            ) as response:
                elapsed = time.time() - start_time
                if response.status in [200, 201]:
                    return True, elapsed, ""
                else:
                    text = await response.text()
                    return False, elapsed, f"HTTP {response.status}: {text[:100]}"
        except Exception as e:
            elapsed = time.time() - start_time
            return False, elapsed, str(e)
    
    async def simulate_user_session(self, user_id: int, duration_seconds: int = 60) -> TestResult:
        """Simulate a single user session"""
        result = TestResult()
        session_id = f"stress_user_{user_id}"
        end_time = time.time() + duration_seconds
        
        while time.time() < end_time:
            # Test getting state (most common operation)
            success, elapsed, error = await self.test_get_state(session_id)
            
            if success:
                result.success_count += 1
            else:
                result.error_count += 1
                result.errors.append(f"get_state: {error}")
            
            result.total_time += elapsed
            result.min_time = min(result.min_time, elapsed)
            result.max_time = max(result.max_time, elapsed)
            
            # Occasionally submit a response (10% of requests)
            if result.success_count % 10 == 0:
                success, elapsed, error = await self.test_submit_response(session_id)
                if success:
                    result.success_count += 1
                else:
                    result.error_count += 1
                    result.errors.append(f"submit_response: {error}")
                
                result.total_time += elapsed
                result.min_time = min(result.min_time, elapsed)
                result.max_time = max(result.max_time, elapsed)
            
            # Wait between requests (simulate human behavior)
            await asyncio.sleep(0.2)  # 5 requests per second per user
        
        return result
    
    async def run_stress_test(self, duration_seconds: int = 60) -> Dict[str, Any]:
        """Run the stress test with multiple concurrent users"""
        print(f"Starting stress test with {self.concurrent_users} concurrent users for {duration_seconds}s")
        print(f"Target URL: {self.base_url}")
        
        start_time = time.time()
        
        # Create tasks for all users
        tasks = [
            self.simulate_user_session(user_id, duration_seconds)
            for user_id in range(self.concurrent_users)
        ]
        
        # Run all user sessions concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_time = time.time() - start_time
        
        # Aggregate results
        total_success = 0
        total_errors = 0
        total_request_time = 0.0
        min_time = float('inf')
        max_time = 0.0
        all_errors = []
        
        for result in results:
            if isinstance(result, Exception):
                all_errors.append(f"User session failed: {result}")
                continue
                
            total_success += result.success_count
            total_errors += result.error_count
            total_request_time += result.total_time
            min_time = min(min_time, result.min_time)
            max_time = max(max_time, result.max_time)
            all_errors.extend(result.errors[:5])  # Limit errors per user
        
        total_requests = total_success + total_errors
        success_rate = (total_success / total_requests * 100) if total_requests > 0 else 0
        avg_response_time = (total_request_time / total_requests) if total_requests > 0 else 0
        requests_per_second = total_requests / total_time
        
        return {
            "duration_seconds": total_time,
            "concurrent_users": self.concurrent_users,
            "total_requests": total_requests,
            "successful_requests": total_success,
            "failed_requests": total_errors,
            "success_rate_percent": success_rate,
            "requests_per_second": requests_per_second,
            "avg_response_time_ms": avg_response_time * 1000,
            "min_response_time_ms": min_time * 1000 if min_time != float('inf') else 0,
            "max_response_time_ms": max_time * 1000,
            "sample_errors": all_errors[:10]  # First 10 errors
        }

async def main():
    parser = argparse.ArgumentParser(description="Stress test the crowdsourcing backend")
    parser.add_argument("--url", default="https://ztclab-1.tail503d36.ts.net", help="Backend URL")
    parser.add_argument("--users", type=int, default=50, help="Number of concurrent users")
    parser.add_argument("--duration", type=int, default=60, help="Test duration in seconds")
    parser.add_argument("--tunnel-url", help="Use cloudflared tunnel URL")
    
    args = parser.parse_args()
    
    # Determine the URL to test
    if args.tunnel_url:
        base_url = args.tunnel_url
    else:
        base_url = args.url
    
    async with BackendStressTester(base_url, args.users) as tester:
        results = await tester.run_stress_test(args.duration)
        
        print("\n" + "="*60)
        print("STRESS TEST RESULTS")
        print("="*60)
        print(f"Duration: {results['duration_seconds']:.1f}s")
        print(f"Concurrent Users: {results['concurrent_users']}")
        print(f"Total Requests: {results['total_requests']}")
        print(f"Successful Requests: {results['successful_requests']}")
        print(f"Failed Requests: {results['failed_requests']}")
        print(f"Success Rate: {results['success_rate_percent']:.1f}%")
        print(f"Requests/Second: {results['requests_per_second']:.1f}")
        print(f"Avg Response Time: {results['avg_response_time_ms']:.1f}ms")
        print(f"Min Response Time: {results['min_response_time_ms']:.1f}ms")
        print(f"Max Response Time: {results['max_response_time_ms']:.1f}ms")
        
        if results['sample_errors']:
            print(f"\nSample Errors:")
            for error in results['sample_errors']:
                print(f"  • {error}")
        
        print("\n" + "="*60)
        
        # Performance assessment
        if results['success_rate_percent'] < 95:
            print("⚠️  WARNING: High error rate detected!")
        if results['avg_response_time_ms'] > 1000:
            print("⚠️  WARNING: High response times detected!")
        if results['requests_per_second'] < args.users * 2:
            print("⚠️  WARNING: Low throughput detected!")
        
        if (results['success_rate_percent'] >= 95 and 
            results['avg_response_time_ms'] < 500 and 
            results['requests_per_second'] >= args.users * 2):
            print("✅ Backend performance looks good!")

if __name__ == "__main__":
    asyncio.run(main())
