import * as THREE        from 'three';
// ---- CHANGE 1: Import OrbitControls ----
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import URDFLoader from 'urdf-loader';

/* ---- scene boilerplate ---- */
const scene   = new THREE.Scene();
scene.background = new THREE.Color(0xeeeeee);

const camera  = new THREE.PerspectiveCamera(45, innerWidth/innerHeight, 0.1, 100);
// ---- CHANGE 2: Move the camera further away to ensure the robot is visible ----
camera.position.set(2, 2, 2); 
scene.add(camera); // It's good practice to add the camera to the scene

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(innerWidth, innerHeight);
document.body.appendChild(renderer.domElement);

// ---- CHANGE 3: Add OrbitControls ----
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 0); // Make the controls look at the center of the scene
controls.update(); // Must be called after any manual changes to the camera's transform

/* basic lighting */
const dirLight = new THREE.DirectionalLight(0xffffff, 10);
dirLight.position.set(3, 4, 5); // Repositioned light slightly
scene.add(dirLight);

scene.add(new THREE.AmbientLight(0xffffff, 1.5)); // Increased ambient light a bit

// ---- CHANGE 4: Add a visual helper to see the center of the scene ----
const gridHelper = new THREE.GridHelper(10, 10);
scene.add(gridHelper);
const axesHelper = new THREE.AxesHelper(1); // X=red, Y=green, Z=blue
scene.add(axesHelper);

async function getInitialJointPositions() {
  try {
    const response = await fetch('http://127.0.0.1:5000/get_initial_joint_positions');
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const jointPositions = await response.json();
    console.log("Successfully fetched initial joint positions:", jointPositions);
    return jointPositions;
  } catch (error) {
    console.error("Could not fetch initial joint positions:", error);
    return {};
  }
}

const manager = new THREE.LoadingManager();
const loader = new URDFLoader(manager);
loader.packages = {
    trossen_arm_description: '/static/trossen_arm_description'
};

loader.load(
  '/static/wxai_base.urdf',
  robot => {
    console.log("Robot loaded successfully:", robot);
    scene.add(robot);

    // robot.rotation.z = Math.PI;
    robot.rotation.x = - Math.PI / 2;

    getInitialJointPositions().then(initialPositions => {
      console.log('Initial positions to be applied:', initialPositions);
      // We will apply these in the next step.
    });
  }
);

/* ---- render loop ---- */
(function animate(){
  requestAnimationFrame(animate);

  // ---- CHANGE 5: Update controls in the render loop ----
  controls.update();

  renderer.render(scene, camera);
})();

// ---- CHANGE 6: Add a resize listener ----
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}, false);