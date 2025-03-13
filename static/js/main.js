document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const uploadForm = document.getElementById('upload-form');
    const fileInput = document.getElementById('image-input');
    const fileName = document.getElementById('file-name');
    const resultSection = document.getElementById('result-section');
    const processing = document.getElementById('processing');
    const results = document.getElementById('results');
    const originalImage = document.getElementById('original-image');
    const downloadBtn = document.getElementById('download-btn');
    const errorMessage = document.getElementById('error-message');
    const errorText = document.getElementById('error-text');

    // Variables for 3D viewer
    let scene, camera, renderer, controls, model;

    // Update file name when file is selected
    fileInput.addEventListener('change', function() {
        if (this.files && this.files[0]) {
            fileName.textContent = this.files[0].name;

            // Validate file type
            const fileType = this.files[0].type;
            if (!fileType.match('image.*')) {
                showError('Please select a valid image file.');
                this.value = '';
                fileName.textContent = 'No file chosen';
                return;
            }

            // Validate file size (max 16MB)
            if (this.files[0].size > 16 * 1024 * 1024) {
                showError('File size exceeds 16MB limit.');
                this.value = '';
                fileName.textContent = 'No file chosen';
                return;
            }
        } else {
            fileName.textContent = 'No file chosen';
        }
    });

    // Handle form submission
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();

        // Validate file input
        if (!fileInput.files || !fileInput.files[0]) {
            showError('Please select an image file.');
            return;
        }

        const formData = new FormData();
        formData.append('image', fileInput.files[0]);

        // Show processing indicator
        resultSection.style.display = 'block';
        processing.style.display = 'flex';
        results.style.display = 'none';
        errorMessage.style.display = 'none';

        // Send the request
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            return response.json().then(data => {
                if (!response.ok) {
                    throw new Error(data.error || 'An error occurred');
                }
                return data;
            });
        })
        .then(data => {
            // Hide processing and show results
            processing.style.display = 'none';
            results.style.display = 'block';

            // Update the image and download link
            originalImage.src = '/' + data.image;
            downloadBtn.href = data.download_url;

            // Initialize 3D viewer with the OBJ model
            init3DViewer(data.obj);
        })
        .catch(error => {
            console.error("Error:", error);
            processing.style.display = 'none';
            showError(error.message || 'Failed to process the image. Please try a different image.');
        });
    });

    // Initialize Three.js 3D viewer
    function init3DViewer(objPath) {
        const container = document.getElementById('3d-preview');

        // Clear container if already initialized
        if (renderer) {
            container.removeChild(renderer.domElement);
        }

        // Create scene
        scene = new THREE.Scene();
        scene.background = new THREE.Color(0xf4f4f8);

        // Create camera
        camera = new THREE.PerspectiveCamera(
            45,
            container.clientWidth / container.clientHeight,
            0.1,
            1000
        );
        camera.position.z = 200;

        // Create renderer
        renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(container.clientWidth, container.clientHeight);
        container.appendChild(renderer.domElement);

        // Add controls - Fixed: Use OrbitControls constructor properly
        controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.25;
        controls.autoRotate = true;
        controls.autoRotateSpeed = 1.0;

        // Add lights
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(1, 1, 1);
        scene.add(directionalLight);

        const backLight = new THREE.DirectionalLight(0xffffff, 0.3);
        backLight.position.set(-1, -1, -1);
        scene.add(backLight);

        // Load OBJ model
        const loader = new THREE.OBJLoader();
        loader.load(
            '/' + objPath,
            function(object) {
                // If previous model exists, remove it
                if (model) {
                    scene.remove(model);
                }

                model = object;

                // Center the model
                const box = new THREE.Box3().setFromObject(model);
                const center = box.getCenter(new THREE.Vector3());
                const size = box.getSize(new THREE.Vector3());

                // Scale model to fit view
                const maxDim = Math.max(size.x, size.y, size.z);
                const scale = 100 / maxDim;
                model.scale.set(scale, scale, scale);

                // Center model
                model.position.sub(center.multiplyScalar(scale));

                // Apply material
                model.traverse(function(child) {
                    if (child.isMesh) {
                        child.material = new THREE.MeshPhongMaterial({
                            color: 0x3f51b5,
                            shininess: 100,
                            specular: 0x111111
                        });
                    }
                });

                scene.add(model);

                // Adjust camera position
                const distance = maxDim * 2;
                camera.position.set(distance, distance / 2, distance);
                camera.lookAt(0, 0, 0);

                // Start animation
                animate();

                // Stop auto-rotation after 5 seconds
                setTimeout(() => {
                    controls.autoRotate = false;
                }, 5000);
            },
            function(xhr) {
                // Progress indicator if needed
                console.log((xhr.loaded / xhr.total * 100) + '% loaded');
            },
            function(error) {
                console.error('Error loading OBJ model', error);
                showError('Failed to load 3D model');
            }
        );

        // Handle window resize
        window.addEventListener('resize', onWindowResize, false);

        function onWindowResize() {
            camera.aspect = container.clientWidth / container.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(container.clientWidth, container.clientHeight);
        }

        function animate() {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }
    }

    // Show error message
    function showError(message) {
        resultSection.style.display = 'block';
        processing.style.display = 'none';
        results.style.display = 'none';
        errorMessage.style.display = 'block';
        errorText.textContent = message;
    }
});