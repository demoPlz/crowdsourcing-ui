import * as THREE        from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

/* ---- scene boilerplate ---- */
const scene   = new THREE.Scene();
scene.background = new THREE.Color(0xeeeeee);

const camera  = new THREE.PerspectiveCamera(45, innerWidth/innerHeight, 0.1, 100);
camera.position.set(2, 1, 2);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(innerWidth, innerHeight);
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);

/* ---- demo geometry: a blue cube ---- */
const cube = new THREE.Mesh(
  new THREE.BoxGeometry(1, 1, 1),
  new THREE.MeshStandardMaterial({ color: 0x55aaff })
);
scene.add(cube);

/* basic lighting */
scene.add(
  new THREE.DirectionalLight(0xffffff, 0.8).position.set(1,2,3),
  new THREE.AmbientLight(0xffffff, 0.4)
);

/* ---- render loop ---- */
(function animate(){
  requestAnimationFrame(animate);
  renderer.render(scene, camera);
})();
