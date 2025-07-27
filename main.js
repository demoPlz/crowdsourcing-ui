import * as THREE        from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { DragControls }  from 'three/examples/jsm/controls/DragControls.js';
import URDFLoader from 'urdf-loader';

/* ---- scene boilerplate ---- */
const scene   = new THREE.Scene();
scene.background = new THREE.Color(0xeeeeee);

const camera  = new THREE.PerspectiveCamera(45, innerWidth/innerHeight, 0.1, 100);
camera.position.set(1, 1, 1);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(innerWidth, innerHeight);
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);

/* basic lighting */
const dirLight = new THREE.DirectionalLight(0xffffff, 10);
dirLight.position.set(1, 2, 3);
scene.add(dirLight);

scene.add(new THREE.AmbientLight(0xffffff, 1));

const manager = new THREE.LoadingManager();
const loader = new URDFLoader( manager );
loader.packages = {
    trossen_arm_description: '/trossen_arm_description/'
};
loader.load(
  '/wxai_base.urdf',
  robot => {

    scene.add( robot );

  }
);

/* ---- render loop ---- */
(function animate(){
  requestAnimationFrame(animate);
  renderer.render(scene, camera);
})();
