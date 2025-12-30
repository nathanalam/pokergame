
// State Variables
let video = null;
let canvasOutput = null;
let maskCanvas = null;
let stream = null;
let vc = null;
let cap = null;

// Optical Flow Globals
let oldGray = null;
let newGray = null;
let p0 = null;
let p1 = null;
let st = null;
let err = null;
let mask = null; // for specifying ROI to find features
let zeroEle = null; // for visualization mask (not used heavily but kept)

// UI State
let isStreaming = false;
let isSelectionStarted = false;
let selectionRect = { x: 0, y: 0, width: 0, height: 0 };
let selectionStart = { x: 0, y: 0 };
let isTracking = false;
let isArmed = false;
let selectionState = 'IDLE';

// Draggable Logic
let isDraggingZone = false;
let dragZoneTarget = null;
let dragOffsets = { x: 0, y: 0 };

// Track center point global
let currentCenter = { x: 0, y: 0 };
let trackBox = { x: 0, y: 0, width: 0, height: 0 };

// Audio
const successAudio = new Audio('https://www.myinstants.com/media/sounds/romanceeeeeeeeeeeeee.mp3');
const failAudio = new Audio('https://www.myinstants.com/media/sounds/tf_nemesis.mp3');

// Called by OpenCV async loader
function onOpenCvReady() {
    console.log('OpenCV.js matches loaded.');
    document.getElementById('loader').classList.add('hidden');
    document.getElementById('app-container').classList.remove('hidden');
    startCamera();
}

function startCamera() {
    video = document.getElementById('webcam');
    canvasOutput = document.getElementById('canvas-output');
    maskCanvas = document.getElementById('mask-output');

    const constraints = {
        video: {
            width: { ideal: 640 },
            height: { ideal: 480 },
            facingMode: "user"
        },
        audio: false
    };

    navigator.mediaDevices.getUserMedia(constraints)
        .then(function (s) {
            stream = s;
            video.srcObject = stream;
            video.play();

            video.oncanplay = function () {
                if (!isStreaming) {
                    initOpenCV();
                    isStreaming = true;
                }
            };
        })
        .catch(function (err) {
            console.error("Camera Error: " + err);
            alert("Camera needed for vision system.");
        });
}

function initOpenCV() {
    if (!video || video.videoWidth === 0 || video.videoHeight === 0) {
        console.warn("Video dimensions not ready yet. Retrying...");
        requestAnimationFrame(initOpenCV);
        return;
    }

    // CRITICAL FIX: OpenCV.js VideoCapture reads width/height attributes
    // We must sync them to the actual stream dimensions
    video.width = video.videoWidth;
    video.height = video.videoHeight;

    const width = video.videoWidth;
    const height = video.videoHeight;

    // Resize Canvas to match Video
    canvasOutput.width = width;
    canvasOutput.height = height;

    // Cleanup
    if (oldGray) oldGray.delete();
    if (newGray) newGray.delete();
    if (p0) p0.delete();
    if (p1) p1.delete();
    if (st) st.delete();
    if (err) err.delete();
    if (mask) mask.delete();

    // Init Mats
    try {
        cap = new cv.VideoCapture(video);

        // We need 2 frames for Optical Flow
        oldGray = new cv.Mat(height, width, cv.CV_8UC1);
        newGray = new cv.Mat(height, width, cv.CV_8UC1);

        // These hold the points
        p0 = new cv.Mat(); // Previous points
        p1 = new cv.Mat(); // New points
        st = new cv.Mat(); // Status 
        err = new cv.Mat(); // Error

        // Mask for creating ROI for 'goodFeaturesToTrack'
        mask = new cv.Mat(height, width, cv.CV_8UC1);

        console.log(`OpenCV Initialized: ${width}x${height} `);

        if (!isStreaming) {
            isStreaming = true;
            requestAnimationFrame(processFrame);
        }

        // Read first frame to ensure oldGray is populated
        let frameMap = new cv.Mat(height, width, cv.CV_8UC4);
        cap.read(frameMap);
        cv.flip(frameMap, frameMap, 1);
        cv.cvtColor(frameMap, oldGray, cv.COLOR_RGBA2GRAY);
        frameMap.delete();

    } catch (e) {
        console.error("OpenCV Init Error:", e);
    }

    setupInputHandlers();
}

function startTracking() {
    if (selectionRect.width <= 0 || selectionRect.height <= 0) return;

    // 1. Define ROI Mask
    // We only look for features INSIDE the box the user drew
    mask.setTo(new cv.Scalar(0)); // Black out everything

    let roiRect = new cv.Rect(selectionRect.x, selectionRect.y, selectionRect.width, selectionRect.height);
    let roi = mask.roi(roiRect);
    roi.setTo(new cv.Scalar(255)); // White inside box
    roi.delete(); // allow Matrix data to remain, just delete view

    // 2. Detect "Good Features to Track" (Corners/High Contrast)
    // maxCorners: 100, qualityLevel: 0.3, minDistance: 7, blockSize: 7
    try {
        // Ensure oldGray is fresh
        cv.cvtColor(src, oldGray, cv.COLOR_RGBA2GRAY);

        // maxCorners: 100, qualityLevel: 0.01 (very sensitive), minDistance: 5
        cv.goodFeaturesToTrack(oldGray, p0, 100, 0.01, 5, mask, 3);

        if (p0.rows === 0) {
            alert("No features found. Try selecting an area with more contrast.");
            return;
        }

        trackBox = { ...selectionRect };
        isTracking = true;

        document.getElementById('status-message').innerText = `Locked ${p0.rows} points.`;
        document.getElementById('status-message').className = "status-ok";
        document.getElementById('start-btn').disabled = false;
        document.getElementById('start-btn').focus();

    } catch (e) {
        console.error("Feature detection error:", e);
    }
}

function processFrame() {
    if (!isStreaming) return;

    try {
        // src is already initialized in initOpenCV and reused
        cap.read(src);
        cv.flip(src, src, 1);

        // Convert to Gray
        cv.cvtColor(src, newGray, cv.COLOR_RGBA2GRAY);

        if (isTracking) {
            // Calculate Optical Flow
            // p0 = old points, p1 = new points calculated
            if (p0.rows > 0) {
                // winSize: (15,15), maxLevel: 2, criteria: (COUNT+EPS, 10, 0.03)
                let winSize = new cv.Size(15, 15);
                let maxLevel = 2;
                let criteria = new cv.TermCriteria(cv.TermCriteria_EPS | cv.TermCriteria_COUNT, 10, 0.03);

                cv.calcOpticalFlowPyrLK(oldGray, newGray, p0, p1, st, err, winSize, maxLevel, criteria);

                // Select good points
                let goodNew = [];
                let dx_sum = 0;
                let dy_sum = 0;
                let count = 0;

                for (let i = 0; i < st.rows; i++) {
                    if (st.data[i] === 1) { // 1 = Found
                        let nx = p1.data32F[i * 2];
                        let ny = p1.data32F[i * 2 + 1];
                        let ox = p0.data32F[i * 2];
                        let oy = p0.data32F[i * 2 + 1];

                        // Check if point moved crazy amount (outlier)
                        let dist = Math.sqrt((nx - ox) * (nx - ox) + (ny - oy) * (ny - oy));
                        if (dist < 50) {
                            goodNew.push(new cv.Point(nx, ny));
                            dx_sum += (nx - ox);
                            dy_sum += (ny - oy);
                            count++;

                            // Draw Tracking Points (Digital Velcro)
                            cv.circle(src, new cv.Point(nx, ny), 3, new cv.Scalar(0, 255, 255, 255), -1);
                        }
                    }
                }

                if (count < 4) {
                    // Lost tracking
                    isTracking = false;
                    document.getElementById('status-message').innerText = "Lost Tracking.";
                    document.getElementById('status-message').className = "status-warn";
                    isArmed = false;
                } else {
                    // Update Box Position based on Average Movement
                    let dx = dx_sum / count;
                    let dy = dy_sum / count;

                    trackBox.x += dx;
                    trackBox.y += dy;

                    currentCenter.x = trackBox.x + trackBox.width / 2;
                    currentCenter.y = trackBox.y + trackBox.height / 2;

                    // Update p0 for next frame
                    p0.delete();
                    p0 = new cv.Mat(goodNew.length, 1, cv.CV_32FC2);
                    for (let i = 0; i < goodNew.length; i++) {
                        p0.data32F[i * 2] = goodNew[i].x;
                        p0.data32F[i * 2 + 1] = goodNew[i].y;
                    }

                    // Draw Box
                    let p1_box = new cv.Point(trackBox.x, trackBox.y);
                    let p2_box = new cv.Point(trackBox.x + trackBox.width, trackBox.y + trackBox.height);
                    cv.rectangle(src, p1_box, p2_box, new cv.Scalar(0, 255, 0, 255), 2);
                    cv.circle(src, currentCenter, 5, new cv.Scalar(0, 0, 255, 255), -1);

                    if (isArmed) checkZones(currentCenter);
                }
            }
        }
        else if (selectionState === 'SELECTING' || selectionState === 'CONFIRMING') {
            // ALWAYS Draw Selection Box if we are selecting
            let color = new cv.Scalar(255, 0, 0, 255);
            let p1_sel = new cv.Point(selectionRect.x, selectionRect.y);
            let p2_sel = new cv.Point(selectionRect.x + selectionRect.width, selectionRect.y + selectionRect.height);
            cv.rectangle(src, p1_sel, p2_sel, color, 2);
        }

        cv.imshow('canvas-output', src);

        // Critical: Update oldGray
        newGray.copyTo(oldGray);

    } catch (e) {
        console.error("Loop Error:", e);
    }

    requestAnimationFrame(processFrame);
}

function setupInputHandlers() {
    const canvas = document.getElementById('canvas-output');

    // Need to handle coordinate mapping from displayed CSS size to internal Canvas size
    function getMousePos(evt) {
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        return {
            x: (evt.clientX - rect.left) * scaleX,
            y: (evt.clientY - rect.top) * scaleY
        };
    }

    // Draggable Zones Setup
    const zones = document.querySelectorAll('.zone');
    zones.forEach(zone => {
        zone.addEventListener('mousedown', (e) => {
            e.stopPropagation(); // Don't trigger canvas selection
            isDraggingZone = true;
            dragZoneTarget = zone;
            const rect = zone.getBoundingClientRect();
            dragOffsets.x = e.clientX - rect.left;
            dragOffsets.y = e.clientY - rect.top;
        });

        // Touch support for zones
        zone.addEventListener('touchstart', (e) => {
            e.stopPropagation();
            e.preventDefault();
            const touch = e.touches[0];
            isDraggingZone = true;
            dragZoneTarget = zone;
            const rect = zone.getBoundingClientRect();
            dragOffsets.x = touch.clientX - rect.left;
            dragOffsets.y = touch.clientY - rect.top;
        }, { passive: false });
    });

    // Global Drag Handlers
    document.addEventListener('mousemove', (e) => {
        if (isDraggingZone && dragZoneTarget) {
            dragZoneTarget.style.left = (e.clientX - dragOffsets.x) + 'px';
            dragZoneTarget.style.top = (e.clientY - dragOffsets.y) + 'px';
            dragZoneTarget.style.right = 'auto'; // Clear potential css right align
        }
    });

    document.addEventListener('mouseup', () => {
        isDraggingZone = false;
        dragZoneTarget = null;
    });

    // Global Touch Move/End
    document.addEventListener('touchmove', (e) => {
        if (isDraggingZone && dragZoneTarget) {
            e.preventDefault();
            const touch = e.touches[0];
            dragZoneTarget.style.left = (touch.clientX - dragOffsets.x) + 'px';
            dragZoneTarget.style.top = (touch.clientY - dragOffsets.y) + 'px';
            dragZoneTarget.style.right = 'auto';
        }
    }, { passive: false });

    document.addEventListener('touchend', () => {
        isDraggingZone = false;
        dragZoneTarget = null;
    });

    // Valid check for setupInputHandlers to not duplicate
    // ...

    // Selection State Machine
    let selectionState = 'IDLE'; // IDLE, SELECTING, CONFIRMING

    canvas.addEventListener('mousedown', (e) => {
        if (isDraggingZone || isArmed || isTracking) return;

        const pos = getMousePos(e);

        if (selectionState === 'IDLE') {
            // First Click: Start Selection
            selectionState = 'SELECTING';
            isSelectionStarted = true; // For drawing loop
            selectionStart = pos;
            selectionRect = { x: pos.x, y: pos.y, width: 0, height: 0 };
            document.getElementById('status-message').innerText = "Click second point to finish box...";
            document.getElementById('status-message').className = "status-warn";

        } else if (selectionState === 'SELECTING') {
            // Second Click: End Selection & Show Confirm
            selectionState = 'CONFIRMING';
            // Finalize rect size
            selectionRect.width = pos.x - selectionStart.x;
            selectionRect.height = pos.y - selectionStart.y;
            normalizeSelectionRect();

            if (selectionRect.width > 5 && selectionRect.height > 5) {
                showConfirmationModal();
            } else {
                // Reset if too small
                cancelSelection();
            }
        }
    });

    canvas.addEventListener('mousemove', (e) => {
        if (isDraggingZone) return;

        if (selectionState === 'SELECTING') {
            const pos = getMousePos(e);
            selectionRect.width = pos.x - selectionStart.x;
            selectionRect.height = pos.y - selectionStart.y;
        }
    });

    // Handle Confirm Modal Buttons
    document.getElementById('confirm-yes').addEventListener('click', () => {
        document.getElementById('confirm-modal').classList.add('hidden');
        selectionState = 'IDLE';
        startTracking();
    });

    document.getElementById('confirm-no').addEventListener('click', () => {
        cancelSelection();
    });

    function normalizeSelectionRect() {
        if (selectionRect.width < 0) {
            selectionRect.x += selectionRect.width;
            selectionRect.width = Math.abs(selectionRect.width);
        }
        if (selectionRect.height < 0) {
            selectionRect.y += selectionRect.height;
            selectionRect.height = Math.abs(selectionRect.height);
        }
    }

    function cancelSelection() {
        document.getElementById('confirm-modal').classList.add('hidden');
        selectionState = 'IDLE';
        isSelectionStarted = false; // Flag for drawing loop
        selectionRect = { x: 0, y: 0, width: 0, height: 0 };
        document.getElementById('status-message').innerText = "Selection cancelled. Try again.";
        document.getElementById('status-message').className = "status-neutral";
    }

    function showConfirmationModal() {
        document.getElementById('confirm-modal').classList.remove('hidden');

        // Generate Preview
        // 1. Calc Hist for the selection
        let roi = src.roi(selectionRect);
        let hsvRoi = new cv.Mat();
        cv.cvtColor(roi, hsvRoi, cv.COLOR_RGBA2RGB);
        cv.cvtColor(hsvRoi, hsvRoi, cv.COLOR_RGB2HSV);

        // 2. Filter
        let maskRoi = new cv.Mat();
        let low = new cv.Mat(hsvRoi.rows, hsvRoi.cols, hsvRoi.type(), [0, 60, 32, 0]);
        let high = new cv.Mat(hsvRoi.rows, hsvRoi.cols, hsvRoi.type(), [180, 255, 255, 0]);
        cv.inRange(hsvRoi, low, high, maskRoi);

        // 3. Show Mask in Preview Canvas
        // We need to display 'maskRoi' on 'preview-canvas'
        cv.imshow('preview-canvas', maskRoi);

        // Cleanup
        roi.delete(); hsvRoi.delete(); maskRoi.delete(); low.delete(); high.delete();
    }

    // Mobile touch support optimization
    canvas.addEventListener('touchstart', (e) => {
        if (isDraggingZone) return;
        const touch = e.touches[0];
        const me = new MouseEvent("mousedown", { clientX: touch.clientX, clientY: touch.clientY });
        canvas.dispatchEvent(me);
    }, { passive: false });
    canvas.addEventListener('touchmove', (e) => {
        if (isDraggingZone) return;
        e.preventDefault(); // Stop scroll
        const touch = e.touches[0];
        const me = new MouseEvent("mousemove", { clientX: touch.clientX, clientY: touch.clientY });
        canvas.dispatchEvent(me);
    }, { passive: false });
    canvas.addEventListener('touchend', (e) => {
        if (isDraggingZone) return;
        const me = new MouseEvent("mouseup", {}); // Mouseup is not used in the new state machine for selection, but keep for consistency if other parts rely on it.
        canvas.dispatchEvent(me);
    });

    // Start System Button
    document.getElementById('start-btn').addEventListener('click', () => {
        // Unlock Audio Context
        successAudio.play().then(() => successAudio.pause()).catch(() => { });
        failAudio.play().then(() => failAudio.pause()).catch(() => { });

        isArmed = true;
        document.getElementById('start-btn').innerText = "Running...";
        document.getElementById('controls').style.opacity = 0.5;
        document.getElementById('active-indicator').classList.remove('hidden');
    });
}

function drawRotatedRect(box, dst) {
    // Determine corners
    let vertices = cv.RotatedRect.points(box);
    let color = new cv.Scalar(0, 255, 0, 255); // Green Tracker
    for (let i = 0; i < 4; i++) {
        cv.line(dst, vertices[i], vertices[(i + 1) % 4], color, 4, cv.LINE_AA, 0);
    }

    // Draw Center
    cv.circle(dst, box.center, 5, new cv.Scalar(0, 0, 255, 255), -1);
}

// Zone Logic
let yesFrames = 0;
let noFrames = 0;

function checkZones(centerPoint) {
    // Map centerPoint (in canvas coords 640x480 typically) to Screen Coords
    // Zones are in Screen Coords (CSS)
    // Actually simpler: Map Zones to Canvas Coords

    const uiYes = document.getElementById('yes-zone').getBoundingClientRect();
    const uiNo = document.getElementById('no-zone').getBoundingClientRect();
    const videoRect = document.getElementById('canvas-output').getBoundingClientRect();

    // Scale Factors
    const scaleX = canvasOutput.width / videoRect.width;
    const scaleY = canvasOutput.height / videoRect.height;

    // Helper to get zone in canvas coords
    function getZoneRect(uiRect) {
        return {
            x: (uiRect.left - videoRect.left) * scaleX,
            y: (uiRect.top - videoRect.top) * scaleY,
            w: uiRect.width * scaleX,
            h: uiRect.height * scaleY
        };
    }

    const yesZone = getZoneRect(uiYes);
    const noZone = getZoneRect(uiNo);

    const x = centerPoint.x;
    const y = centerPoint.y;

    // Check YES
    if (x >= yesZone.x && x <= (yesZone.x + yesZone.w) &&
        y >= yesZone.y && y <= (yesZone.y + yesZone.h)) {
        yesFrames++;
        noFrames = 0;
        cv.putText(src, `VERIFYING: ${yesFrames} `, { x: 50, y: 50 }, cv.FONT_HERSHEY_PLAIN, 2.0, new cv.Scalar(0, 255, 0, 255), 2);

        if (yesFrames > 30) { // ~1 second @ 30fps
            triggerSuccess();
        }
    }
    // Check NO
    else if (x >= noZone.x && x <= (noZone.x + noZone.w) &&
        y >= noZone.y && y <= (noZone.y + noZone.h)) {
        noFrames++;
        yesFrames = 0;
        cv.putText(src, `FAILING: ${noFrames} `, { x: src.cols - 200, y: 50 }, cv.FONT_HERSHEY_PLAIN, 2.0, new cv.Scalar(255, 0, 0, 255), 2);

        if (noFrames > 15) { // Faster fail
            triggerFail();
        }
    }
    else {
        yesFrames = 0;
        noFrames = 0;
    }
}

function triggerSuccess() {
    isArmed = false;
    successAudio.currentTime = 0;
    successAudio.play();
    document.getElementById('active-overlay').classList.add('visible');
    document.getElementById('message-text').innerText = "YES CONFIRMED";

    const duration = 5 * 1000;
    const end = Date.now() + duration;
    (function frame() {
        confetti({ particleCount: 5, angle: 60, spread: 55, origin: { x: 0 } });
        confetti({ particleCount: 5, angle: 120, spread: 55, origin: { x: 1 } });
        if (Date.now() < end) requestAnimationFrame(frame);
    }());
}

function triggerFail() {
    isArmed = false;
    failAudio.currentTime = 0;
    failAudio.play();
    const toast = document.getElementById('toast');
    toast.classList.add('visible');

    setTimeout(() => {
        toast.classList.remove('visible');
        isArmed = true; // reset
        yesFrames = 0;
        noFrames = 0;
    }, 5000);
}
