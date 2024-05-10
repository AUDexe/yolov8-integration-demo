// File path for the worker script
const WORKER_FILE_PATH = 'worker.js';

// Dimensions for input images expected by the YOLOv8 model
const YOLOV8_INPUT_IMAGE_WIDTH = 640;
const YOLOV8_INPUT_IMAGE_HEIGHT = 640;

// Maximum number of objects that the YOLOv8 model can detect
const YOLOV8_PRETRAINED_MODEL_OBJECT_DETECTION_LIMIT = 80;

// Length of the output array produced by the YOLOv8 model
const YOLOV8_OUTPUT_LENGTH = 8400;

// Class names that the YOLOv8 model can detect
const YOLOV8_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
    'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
];

// Message displayed when access to webcam is denied
const WEBCAM_ACCESS_DENIED_MESSAGE = 'Enabling access to your camera is necessary for us to perform the recognition.';

// Create a new Web Worker instance, specifying the path to the worker script
const worker = new Worker(WORKER_FILE_PATH);

// Flag to indicate whether the worker is currently processing a task
let isWorkerBusy = false;

// Event listener for messages from the worker
worker.onmessage = (event) => {
    // Extract model output from the message event data
    const modelOutput = event.data;

    // Process the YOLOv8 model output to extract detected classes
    const yolov8Class = processYolov8ModelOutput(modelOutput);

    // Log the detected classes to the console
    console.log(yolov8Class);

    // Update the 'isWorkerBusy' flag to indicate that processing is complete
    isWorkerBusy = false;
};

function startWebcamStream() {
    // Request access to the user's webcam
    window.navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            // Get the video element from the HTML
            const video = document.querySelector('#cameraFeed');

            // Set the video element's source to the webcam stream
            video.srcObject = stream;

            // Once the metadata for the video stream is loaded, log the video's dimensions to the console
            video.onloadedmetadata = () => {
                console.log(video.videoWidth, video.videoHeight);

                // Get the canvas element from the HTML
                const canvas = document.querySelector('#cameraStreamCanvas');

                // Set the canvas dimensions to match the video dimensions
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;

                // Get the 2D rendering context of the canvas
                const context = canvas.getContext('2d');

                // Set up an interval to continuously draw frames from the video onto the canvas every 30 milliseconds
                const drawInterval = setInterval(() => {
                    // Draw the current frame of the video onto the canvas at position (0, 0)
                    context.drawImage(video, 0, 0);

                    // Prepare input data for the YOLOv8 model using the canvas
                    const modelInput = prepareInputForYolov8Model(canvas);

                    // Check if the worker is not busy processing a task
                    if (!isWorkerBusy) {
                        // Send the prepared input data to the worker for processing
                        worker.postMessage(modelInput);

                        // Update the 'isWorkerBusy' flag to indicate that the worker is now busy
                        isWorkerBusy = true;
                    }
                }, 30);
            };
        })
        .catch(() => {
            // Alert the user if access to webcam is denied
            alert(WEBCAM_ACCESS_DENIED_MESSAGE);
        });
}

function prepareInputForYolov8Model(image) {
    // Create a temporary canvas and resize it to the specified dimensions
    const canvas = document.createElement('canvas');
    canvas.width = YOLOV8_INPUT_IMAGE_WIDTH;
    canvas.height = YOLOV8_INPUT_IMAGE_HEIGHT;

    // Get the 2D rendering context of the canvas
    const context = canvas.getContext('2d');

    // Draw the received image onto the canvas at position (0, 0) with specified dimensions
    context.drawImage(image, 0, 0, YOLOV8_INPUT_IMAGE_WIDTH, YOLOV8_INPUT_IMAGE_HEIGHT);

    // Extract pixel data from the canvas context for the entire frame
    const pixelData = context.getImageData(0, 0, YOLOV8_INPUT_IMAGE_WIDTH, YOLOV8_INPUT_IMAGE_HEIGHT).data;

    // Arrays to store normalized color channel values (red, green, blue)
    const red = [], green = [], blue = [];

    // Loop through the pixel data array, iterating over each RGBA component (4 values per pixel)
    for (let index = 0; index < pixelData.length; index += 4) {
        // Normalize the red channel value (range 0-255) to range 0-1 and push to the red array
        red.push(pixelData[index] / 255);

        // Normalize the green channel value (range 0-255) to range 0-1 and push to the green array
        green.push(pixelData[index + 1] / 255);

        // Normalize the blue channel value (range 0-255) to range 0-1 and push to the blue array
        blue.push(pixelData[index + 2] / 255);
    }

    // Concatenate the red, green, and blue arrays to a single one in which reds go first, greens go next and blues go last
    return [...red, ...green, ...blue];
}

function processYolov8ModelOutput(output) {
    // Loop through each index up to 8400 (assuming this is the total number of outputs)
    for (let index = 0; index < YOLOV8_OUTPUT_LENGTH; index++) {
        // Find the class with the highest probability at the given index
        const [class_id, prob] = [...Array(YOLOV8_PRETRAINED_MODEL_OBJECT_DETECTION_LIMIT).keys()]
            .map(col => [col, output[YOLOV8_OUTPUT_LENGTH * (col + 4) + index]])
            .reduce((accum, item) => item[1] > accum[1] ? item : accum, [0, 0]);

        // If the probability is less than 0.5, skip to the next index
        if (prob < 0.5) {
            continue;
        }

        // Return the class name corresponding to the highest probability
        return YOLOV8_CLASSES[class_id];
    }
}

// Call the function to start the webcam stream
startWebcamStream();