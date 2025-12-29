
document.addEventListener('DOMContentLoaded', () => {
    const video = document.getElementById('webcam');
    const canvas = document.getElementById('tracking-canvas');
    const context = canvas.getContext('2d');
    const startBtn = document.getElementById('start-btn');
    const controls = document.getElementById('controls');
    const zones = document.querySelectorAll('.zone');
    const yesZone = document.getElementById('yes-zone');
    const noZone = document.getElementById('no-zone');
    const sampleDisplay = document.getElementById('color-sample-display');
    const statusIndicator = document.getElementById('status-indicator');
    const activeOverlay = document.getElementById('active-overlay');
    const messageText = document.getElementById('message-text');
    const toast = document.getElementById('toast');

    // Audio
    const successAudio = new Audio('https://www.myinstants.com/media/sounds/romanceeeeeeeeeeeeee.mp3');
    const failAudio = new Audio('https://www.myinstants.com/media/sounds/tf_nemesis.mp3');

    // State
    let isArmed = false;
    let targetColor = { r: 0, g: 0, b: 0 };
    let tracker = null;
    let trackingTask = null;
    let yesFrameCount = 0;
    let noFrameCount = 0;
    const FRAME_THRESHOLD = 10;

    // Draggable Logic
    let isDragging = false;
    let currentDragInfo = null;

    zones.forEach(zone => {
        zone.addEventListener('mousedown', (e) => {
            // Check if clicking resize handle (bottom right)
            const rect = zone.getBoundingClientRect();
            if (e.clientX > rect.right - 20 && e.clientY > rect.bottom - 20) return; // Allow resize

            isDragging = true;
            currentDragInfo = {
                element: zone,
                offsetX: e.clientX - zone.offsetLeft,
                offsetY: e.clientY - zone.offsetTop
            };
        });
    });

    document.addEventListener('mousemove', (e) => {
        if (!isDragging || !currentDragInfo) return;
        const { element, offsetX, offsetY } = currentDragInfo;
        element.style.left = (e.clientX - offsetX) + 'px';
        element.style.top = (e.clientY - offsetY) + 'px';
    });

    document.addEventListener('mouseup', () => {
        isDragging = false;
        currentDragInfo = null;
    });

    // 1. Initialize Camera
    navigator.mediaDevices.getUserMedia({ video: true, audio: false })
        .then(stream => {
            video.srcObject = stream;
        })
        .catch(err => {
            console.error("Camera access denied or failed.", err);
            alert("Camera access is required for this system to function.");
        });

    // 2. Color Sampling
    // tracking.js doesn't give us a direct click-to-sample easy way on the video element itself without a canvas overlay.
    // We will use the canvas to sample.
    function sampleColor(e) {
        if (isArmed) return;

        // Draw current video frame to canvas to sample pixel
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        context.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Get click coordinates relative to video/canvas
        const rect = video.getBoundingClientRect();
        // Calculate logical scaling
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;

        const x = (e.clientX - rect.left) * scaleX;
        const y = (e.clientY - rect.top) * scaleY;

        const pixel = context.getImageData(x, y, 1, 1).data;
        targetColor = { r: pixel[0], g: pixel[1], b: pixel[2] };

        // Visual feedback
        sampleDisplay.style.backgroundColor = `rgb(${targetColor.r}, ${targetColor.g}, ${targetColor.b})`;
        console.log("Sampled Color:", targetColor);
    }

    // Attach click to video element directly as ui-layer is pointer-events: none
    video.addEventListener('click', (e) => {
        // e.target is video.
        sampleColor(e);
    });

    // Helper: RGB to HSV
    function rgbToHsv(r, g, b) {
        r /= 255, g /= 255, b /= 255;
        const max = Math.max(r, g, b), min = Math.min(r, g, b);
        let h, s, v = max;
        const d = max - min;
        s = max === 0 ? 0 : d / max;

        if (max === min) {
            h = 0; // achromatic
        } else {
            switch (max) {
                case r: h = (g - b) / d + (g < b ? 6 : 0); break;
                case g: h = (b - r) / d + 2; break;
                case b: h = (r - g) / d + 4; break;
            }
            h /= 6;
        }
        return [h, s, v];
    }

    // 3. Register Custom Color Tracker
    tracking.ColorTracker.registerColor('custom', function (r, g, b) {
        if (targetColor.r === 0 && targetColor.g === 0 && targetColor.b === 0) return false;

        const [th, ts, tv] = rgbToHsv(targetColor.r, targetColor.g, targetColor.b);
        const [h, s, v] = rgbToHsv(r, g, b);

        // Tolerances
        // Hue: 0-1, circular. Allow e.g. 0.1 (36 degrees) diff.
        // Saturation: 0-1. Allow e.g. 0.3 diff.
        // Value: 0-1. Allow large diff e.g. 0.5 for lighting.

        let hueDiff = Math.abs(h - th);
        if (hueDiff > 0.5) hueDiff = 1 - hueDiff; // Wrap around

        const sDiff = Math.abs(s - ts);
        const vDiff = Math.abs(v - tv);

        // Strict Hue, Medium Saturation, Loose Value
        return hueDiff < 0.08 && sDiff < 0.35 && vDiff < 0.6;
    });

    // 4. Start System
    startBtn.addEventListener('click', () => {
        if (targetColor.r === 0 && targetColor.g === 0 && targetColor.b === 0) {
            alert("Please sample a color first (Click the ball on screen).");
            return;
        }

        // Unlock Audio
        successAudio.play().then(() => successAudio.pause()).catch(e => console.log(e));
        failAudio.play().then(() => failAudio.pause()).catch(e => console.log(e));

        // UI Changes
        isArmed = true;
        controls.style.display = 'none';
        zones.forEach(z => {
            z.style.borderStyle = 'dashed'; // Visual cue
            z.style.opacity = '0.3'; // Dim them
            z.style.pointerEvents = 'none'; // Lock position
        });
        statusIndicator.style.display = 'block';

        // Start Tracking logic
        startTracking();
    });

    function getZoneRect(zoneElement) {
        const videoRect = video.getBoundingClientRect();
        const zoneRect = zoneElement.getBoundingClientRect();

        // Relative coordinates (0-1) to handle scaling matches
        // Actually, tracking.js returns coordinates in the canvas/video pixel space
        // We need to map screen coordinates of div to video coordinates

        const scaleX = video.videoWidth / videoRect.width;
        const scaleY = video.videoHeight / videoRect.height;

        return {
            x: (zoneRect.left - videoRect.left) * scaleX,
            y: (zoneRect.top - videoRect.top) * scaleY,
            width: zoneRect.width * scaleX,
            height: zoneRect.height * scaleY
        };
    }

    function startTracking() {
        tracker = new tracking.ColorTracker(['custom']);
        tracker.setMinDimension(5); // Minimum size of blob

        trackingTask = tracking.track('#webcam', tracker);

        tracker.on('track', function (event) {
            if (!isArmed) return;

            context.clearRect(0, 0, canvas.width, canvas.height);

            if (event.data.length === 0) {
                // No color found
                return;
            }

            event.data.forEach(function (rect) {
                // Calculate centroid
                const cx = rect.x + rect.width / 2;
                const cy = rect.y + rect.height / 2;

                // Debug draw
                // context.strokeStyle = rect.color;
                // context.strokeRect(rect.x, rect.y, rect.width, rect.height);

                checkZones(cx, cy);
            });
        });
    }

    function checkZones(x, y) {
        const yesRect = getZoneRect(yesZone);
        const noRect = getZoneRect(noZone);

        // Check YES
        if (x >= yesRect.x && x <= yesRect.x + yesRect.width &&
            y >= yesRect.y && y <= yesRect.y + yesRect.height) {

            yesFrameCount++;
            noFrameCount = 0; // Reset opposite

            if (yesFrameCount > FRAME_THRESHOLD) {
                triggerSuccess();
            }
        }
        // Check NO
        else if (x >= noRect.x && x <= noRect.x + noRect.width &&
            y >= noRect.y && y <= noRect.y + noRect.height) {

            noFrameCount++;
            yesFrameCount = 0;

            // Immediate fail or threshold? Let's do small threshold
            if (noFrameCount > 5) {
                triggerFail();
            }
        }
        else {
            // Reset if strictly outside? Maybe decay instead?
            // For now, strict reset to avoid accidental triggers
            yesFrameCount = 0;
            noFrameCount = 0;
        }
    }

    function triggerSuccess() {
        if (!isArmed) return;
        isArmed = false; // Stop logic
        trackingTask.stop();

        successAudio.currentTime = 0;
        successAudio.play();

        activeOverlay.classList.remove('hidden');
        activeOverlay.classList.add('visible');
        messageText.innerText = "SHE SAID YES!";

        // Confetti
        const duration = 5 * 1000;
        const animationEnd = Date.now() + duration;
        const defaults = { startVelocity: 30, spread: 360, ticks: 60, zIndex: 200 };

        const randomInRange = (min, max) => Math.random() * (max - min) + min;

        const interval = setInterval(function () {
            const timeLeft = animationEnd - Date.now();
            if (timeLeft <= 0) {
                return clearInterval(interval);
            }
            const particleCount = 50 * (timeLeft / duration);
            confetti(Object.assign({}, defaults, { particleCount, origin: { x: randomInRange(0.1, 0.3), y: Math.random() - 0.2 } }));
            confetti(Object.assign({}, defaults, { particleCount, origin: { x: randomInRange(0.7, 0.9), y: Math.random() - 0.2 } }));
        }, 250);
    }

    function triggerFail() {
        if (!isArmed) return;
        isArmed = false;

        failAudio.currentTime = 0;
        failAudio.play();

        toast.classList.add('visible');

        setTimeout(() => {
            toast.classList.remove('visible');
            isArmed = true; // Reset
            yesFrameCount = 0;
            noFrameCount = 0;
        }, 5000);
    }

});
