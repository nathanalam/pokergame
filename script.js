
// State Variables
let video = null;
let canvasOutput = null;
let maskCanvas = null;
let stream = null;
let vc = null; // Video Capture
let cap = null; // OpenCV Video Capture
let src = null;
let dst = null;
let hsv = null;
let hue = null;
let mask = null;
let hist = null;
let hsvVec = null;
let termCrit = null;
let trackWindow = null;
let trackBox = null;

// UI State
let isStreaming = false;
let isSelectionStarted = false;
let selectionRect = { x: 0, y: 0, width: 0, height: 0 };
let selectionStart = { x: 0, y: 0 };
let isTracking = false;
let isArmed = false;

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
    const width = video.videoWidth;
    const height = video.videoHeight;
    canvasOutput.width = width;
    canvasOutput.height = height;

    // Initialize Mats
    cap = new cv.VideoCapture(video);
    src = new cv.Mat(height, width, cv.CV_8UC4);
    dst = new cv.Mat(height, width, cv.CV_8UC4);
    hsv = new cv.Mat(height, width, cv.CV_8UC3);
    hue = new cv.Mat(height, width, cv.CV_8UC1);
    mask = new cv.Mat(height, width, cv.CV_8UC1);
    hist = new cv.Mat();
    hsvVec = new cv.MatVector();
    hsvVec.push_back(hsv);

    // Termination criteria for CamShift: (EPS | COUNT, 10 iterations, 1px movement)
    // Basically stop if it moves <1px or hits 10 loops
    termCrit = new cv.TermCriteria(cv.TermCriteria_EPS | cv.TermCriteria_COUNT, 10, 1);

    // Render Loop
    requestAnimationFrame(processFrame);

    // Attach Listeners
    setupInputHandlers();
}

function processFrame() {
    if (!isStreaming) return;

    try {
        cap.read(src); // Read frame from video

        // Mirror effect for UX (flip horizontally)
        cv.flip(src, src, 1);

        if (isTracking) {
            // 1. Convert to HSV
            cv.cvtColor(src, hsv, cv.COLOR_RGBA2RGB);
            cv.cvtColor(hsv, hsv, cv.COLOR_RGB2HSV);

            // 2. Extract Hue (channel 0)
            // But we actually use ranges inBackProject
            // Let's refine for cv.calcBackProject: expects an array of images.
            // We need to calculate BackProjection based on Histogram 'hist'

            // InRange to filter out weak saturation/dark values (noise reduction)
            let low = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [0, 60, 32, 0]);
            let high = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [180, 255, 255, 0]);
            cv.inRange(hsv, low, high, mask);
            low.delete(); high.delete(); // Cleanup temp mats

            // 3. BackProject
            // args: images, channels, hist, dst, ranges, scale
            // channels = [0] (hue)
            let ch = [0];
            // ranges = [0, 180]

            // For CalcBackProject to work accurately in JS, we need Vector
            // hsvVec already has hsv

            // Use just Hue channel for BackProject? 
            // Often standard implementation extracts Hue first.
            // Let's extract Hue Channel 0 to 'hue' Mat
            let vectorOfMats = new cv.MatVector();
            vectorOfMats.push_back(hsv);

            cv.calcBackProject(vectorOfMats, [0], hist, hue, [0, 180], 1);

            // Bitwise AND with mask (saturation/value filter)
            cv.bitwise_and(hue, mask, hue);

            vectorOfMats.delete();

            // 4. CamShift
            // trackWindow is {x, y, width, height}
            // CamShift modifies trackWindow
            [trackBox, trackWindow] = cv.CamShift(hue, trackWindow, termCrit);

            // 5. Draw Results
            // trackBox is RotatedRect { center, size, angle }
            drawRotatedRect(trackBox, src);

            // Visualization for user (Debug Mask)
            cv.imshow('mask-output', hue);

            // GAME LOGIC
            if (isArmed) {
                checkZones(trackBox.center);
            }

        } else if (isSelectionStarted) {
            // Visualize selection box while dragging
            let color = new cv.Scalar(255, 0, 0, 255);
            let p1 = new cv.Point(selectionRect.x, selectionRect.y);
            let p2 = new cv.Point(selectionRect.x + selectionRect.width, selectionRect.y + selectionRect.height);
            cv.rectangle(src, p1, p2, color, 2);
        }

        // Draw final result to canvas
        cv.imshow('canvas-output', src);

    } catch (err) {
        console.error("CV Loop Error:", err);
    }

    requestAnimationFrame(processFrame);
}

function startTracking() {
    // 1. Convert ROI to HSV
    // ROI = [x, y, w, h]
    if (selectionRect.width <= 0 || selectionRect.height <= 0) return;

    let roi = src.roi(selectionRect);
    let hsvRoi = new cv.Mat();
    cv.cvtColor(roi, hsvRoi, cv.COLOR_RGBA2RGB);
    cv.cvtColor(hsvRoi, hsvRoi, cv.COLOR_RGB2HSV);

    // 2. Filter low saturation/brightness
    let maskRoi = new cv.Mat();
    let low = new cv.Mat(hsvRoi.rows, hsvRoi.cols, hsvRoi.type(), [0, 60, 32, 0]);
    let high = new cv.Mat(hsvRoi.rows, hsvRoi.cols, hsvRoi.type(), [180, 255, 255, 0]);
    cv.inRange(hsvRoi, low, high, maskRoi);

    // 3. Calc Histogram
    let roiVec = new cv.MatVector();
    roiVec.push_back(hsvRoi);

    // Channels [0], Mask maskRoi, Hist hist, bins [16] (Quantize hue for better stability), ranges [0, 180]
    // Note: Using fewer bins (16-30) usually stabilizes CamShift better than 180 bins
    cv.calcHist(roiVec, [0], maskRoi, hist, [16], [0, 180]);
    cv.normalize(hist, hist, 0, 255, cv.NORM_MINMAX);

    // 4. Set Initial Window
    trackWindow = new cv.Rect(selectionRect.x, selectionRect.y, selectionRect.width, selectionRect.height);

    // Cleanup
    roi.delete(); hsvRoi.delete(); maskRoi.delete(); low.delete(); high.delete(); roiVec.delete();

    isTracking = true;

    // UI Updates
    document.getElementById('status-message').innerText = "Object Profiled. Tracking Active.";
    document.getElementById('status-message').className = "status-ok";
    document.getElementById('start-btn').disabled = false;
    document.getElementById('start-btn').focus();
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

    canvas.addEventListener('mousedown', (e) => {
        if (isArmed || isTracking) {
            // If clicked while tracking, reset? 
            // Let's allow re-selection
            isTracking = false;
            isArmed = false;
            document.getElementById('status-message').innerText = "Re-selecting...";
        }

        isSelectionStarted = true;
        const pos = getMousePos(e);
        selectionStart = pos;
        selectionRect = { x: pos.x, y: pos.y, width: 0, height: 0 };
    });

    canvas.addEventListener('mousemove', (e) => {
        if (!isSelectionStarted) return;
        const pos = getMousePos(e);
        selectionRect.width = pos.x - selectionStart.x;
        selectionRect.height = pos.y - selectionStart.y;
    });

    canvas.addEventListener('mouseup', (e) => {
        isSelectionStarted = false;

        // Normalize Rect (handle negative width dragging)
        if (selectionRect.width < 0) {
            selectionRect.x += selectionRect.width;
            selectionRect.width = Math.abs(selectionRect.width);
        }
        if (selectionRect.height < 0) {
            selectionRect.y += selectionRect.height;
            selectionRect.height = Math.abs(selectionRect.height);
        }

        if (selectionRect.width > 10 && selectionRect.height > 10) {
            startTracking();
        }
    });

    // Mobile touch support
    canvas.addEventListener('touchstart', (e) => {
        const touch = e.touches[0];
        const me = new MouseEvent("mousedown", { clientX: touch.clientX, clientY: touch.clientY });
        canvas.dispatchEvent(me);
    }, { passive: false });
    canvas.addEventListener('touchmove', (e) => {
        e.preventDefault(); // Stop scroll
        const touch = e.touches[0];
        const me = new MouseEvent("mousemove", { clientX: touch.clientX, clientY: touch.clientY });
        canvas.dispatchEvent(me);
    }, { passive: false });
    canvas.addEventListener('touchend', (e) => {
        const me = new MouseEvent("mouseup", {});
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
        cv.putText(src, `VERIFYING: ${yesFrames}`, { x: 50, y: 50 }, cv.FONT_HERSHEY_PLAIN, 2.0, new cv.Scalar(0, 255, 0, 255), 2);

        if (yesFrames > 30) { // ~1 second @ 30fps
            triggerSuccess();
        }
    }
    // Check NO
    else if (x >= noZone.x && x <= (noZone.x + noZone.w) &&
        y >= noZone.y && y <= (noZone.y + noZone.h)) {
        noFrames++;
        yesFrames = 0;
        cv.putText(src, `FAILING: ${noFrames}`, { x: src.cols - 200, y: 50 }, cv.FONT_HERSHEY_PLAIN, 2.0, new cv.Scalar(255, 0, 0, 255), 2);

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
