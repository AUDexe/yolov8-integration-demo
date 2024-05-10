// File path for the worker script
const WORKER_FILE_PATH = 'worker.js';

// Required Dimensions for YOLOv8 model images
const YOLOV8_IMAGE_WIDTH = 640;
const YOLOV8_IMAGE_HEIGHT = 640;

// Maximum number of objects that the YOLOv8 model can detect
const YOLOV8_PRETRAINED_MODEL_OBJECT_DETECTION_LIMIT = 80;

// Dimensions of the output matrix generated by YOLOv8 model
const YOLOV8_OUTPUT_ROW_COUNT = 84;
const YOLOV8_OUTPUT_COLUMN_COUNT = 8400;

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

// Color for the border of bounding boxes
const BOUNDING_BOXES_BORDER_COLOR = '#00FF00';

// Width of the border for bounding boxes
const BOUNDING_BOXES_BORDER_WIDTH = 3;

// Horizontal padding for the text label within the bounding box
const OUTPUT_YOLOV8_CLASS_HORIZONTAL_PADDING = 10;

// Height of the text area within the bounding box
const OUTPUT_YOLOV8_CLASS_TEXT_HEIGHT = 25;

// Color for the text of the output class label within the bounding box
const OUTPUT_YOLOV8_CLASS_TEXT_COLOR = '#000000';

// Font style for the text of the output class label within the bounding box
const OUTPUT_YOLOV8_CLASS_TEXT_FONT_STYLE = '18px serif';

// Background color for the text label within the bounding box
const OUTPUT_YOLOV8_CLASS_TEXT_BACKGROUND_COLOR = BOUNDING_BOXES_BORDER_COLOR;

// Message displayed when access to webcam is denied
const WEBCAM_ACCESS_DENIED_MESSAGE = 'Enabling access to your camera is necessary for us to perform the recognition.';

// Array to store bounding boxes
let boxes = [];

// Create a new Web Worker instance, specifying the path to the worker script
const worker = new Worker(WORKER_FILE_PATH);

// Flag to indicate whether the worker is currently processing a task
let isWorkerBusy = false;

// Event listener for messages from the worker
worker.onmessage = (event) => {
    // Extract model output from the message event data
    const modelOutput = event.data;

    // Get the canvas element from the HTML
    const canvas = document.querySelector('#cameraStreamCanvas');

    // Process the YOLOv8 model output to extract bounding boxes
    boxes = processYolov8ModelOutput(modelOutput, canvas.width, canvas.height);

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
                setInterval(() => {
                    // Draw the current frame of the video onto the canvas at position (0, 0)
                    context.drawImage(video, 0, 0);

                    // Draw bounding boxes around detected objects on a canvas
                    drawBoxes(canvas, boxes);

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
    canvas.width = YOLOV8_IMAGE_WIDTH;
    canvas.height = YOLOV8_IMAGE_HEIGHT;

    // Get the 2D rendering context of the canvas
    const context = canvas.getContext('2d');

    // Draw the received image onto the canvas at position (0, 0) with specified dimensions
    context.drawImage(image, 0, 0, YOLOV8_IMAGE_WIDTH, YOLOV8_IMAGE_HEIGHT);

    // Extract pixel data from the canvas context for the entire frame
    const pixelData = context.getImageData(0, 0, YOLOV8_IMAGE_WIDTH, YOLOV8_IMAGE_HEIGHT).data;

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

// For better understanding refere to https://dev.to/andreygermanov/how-to-create-yolov8-based-object-detection-web-service-using-python-julia-nodejs-javascript-go-and-rust-4o8e#process_the_output_nodejs
function processYolov8ModelOutput(output, imageWidth, imageHeight) {
    // Array to store bounding boxes
    let boxes = [];

    // Traverse flat array output by columns
    for (let column = 0; column < YOLOV8_OUTPUT_COLUMN_COUNT; column++) {
        // Find the class with the highest probability
        const [class_id, prob] = [...Array(YOLOV8_PRETRAINED_MODEL_OBJECT_DETECTION_LIMIT).keys()]
            .map(probRow => {
                // Offset row index by 4
                let row = probRow + 4;

                // Extract probability for each class
                return [probRow, output[YOLOV8_OUTPUT_COLUMN_COUNT * row + column]];
            })
            // Find the class with the highest probability using reduce
            .reduce((accum, item) => item[1] > accum[1] ? item : accum, [0, 0]);

        // If the probability is less than 0.5, skip to the next column
        if (prob < 0.5) {
            continue;
        }

        // Class name corresponding to the class_id
        const outputClass = YOLOV8_CLASSES[class_id];

        // Extract the center coordinates of the bounding box from the output array
        const centerX = output[column];
        const centerY = output[YOLOV8_OUTPUT_COLUMN_COUNT + column];

        // Extract the width and height of the bounding box from the output array
        const width = output[YOLOV8_OUTPUT_COLUMN_COUNT * 2 + column];
        const height = output[YOLOV8_OUTPUT_COLUMN_COUNT * 3 + column];

        // Calculate the coordinates of the top-left and bottom-right corners of the bounding box
        const x1 = (centerX - width / 2) / YOLOV8_IMAGE_WIDTH * imageWidth;
        const y1 = (centerY - height / 2) / YOLOV8_IMAGE_HEIGHT * imageHeight;
        const x2 = (centerX + width / 2) / YOLOV8_IMAGE_WIDTH * imageWidth;
        const y2 = (centerY + height / 2) / YOLOV8_IMAGE_HEIGHT * imageHeight;

        // Push bounding box info into the boxes array
        boxes.push([x1, y1, x2, y2, outputClass, prob]);
    }

    // Sort bounding boxes based on confidence scores in descending order
    boxes = boxes.sort((box1, box2) => box2[5] - box1[5]);

    // Array to store filtered bounding boxes after applying the Intersection Over Union algorithm to filter out all overlapped boxes
    const filteredBoxes = [];

    // Apply the Intersection Over Union algorithm to filter out all overlapped boxes
    while (boxes.length > 0) {
        // Add the bounding box with the highest confidence score to the filteredBoxes array
        filteredBoxes.push(boxes[0]);

        // Filter out boxes with high Intersection over Union (IoU) with the current box
        boxes = boxes.filter(box => iou(boxes[0], box) < 0.7);
    }

    // Return the final filtered bounding boxes array
    return filteredBoxes;
}

function iou(box1, box2) {
    return intersection(box1, box2) / union(box1, box2);
}

function union(box1, box2) {
    const [box1_x1, box1_y1, box1_x2, box1_y2] = box1;
    const [box2_x1, box2_y1, box2_x2, box2_y2] = box2;
    const box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    const box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    return box1_area + box2_area - intersection(box1, box2)
}

function intersection(box1, box2) {
    const [box1_x1, box1_y1, box1_x2, box1_y2] = box1;
    const [box2_x1, box2_y1, box2_x2, box2_y2] = box2;
    const x1 = Math.max(box1_x1, box2_x1);
    const y1 = Math.max(box1_y1, box2_y1);
    const x2 = Math.min(box1_x2, box2_x2);
    const y2 = Math.min(box1_y2, box2_y2);
    return (x2 - x1) * (y2 - y1)
}

function drawBoxes(canvas, boxes) {
    // Get the 2D rendering context of the canvas
    const context = canvas.getContext('2d');

    // Set the color of the bounding box outlines
    context.strokeStyle = BOUNDING_BOXES_BORDER_COLOR;

    // Set the line width for the bounding box outlines
    context.lineWidth = BOUNDING_BOXES_BORDER_WIDTH;

    // Set the font style for text within the bounding boxes
    context.font = OUTPUT_YOLOV8_CLASS_TEXT_FONT_STYLE;

    // Iterate over each bounding box in the 'boxes' array
    boxes.forEach(([x1, y1, x2, y2, outputClass]) => {
        // Draw the bounding box outline
        context.strokeRect(x1, y1, x2 - x1, y2 - y1);

        // Set the fill color for the text background
        context.fillStyle = OUTPUT_YOLOV8_CLASS_TEXT_BACKGROUND_COLOR;

        // Measure the width of the output class text
        const width = context.measureText(outputClass).width;

        // Draw a filled rectangle as background for the text
        context.fillRect(x1, y1, width + OUTPUT_YOLOV8_CLASS_HORIZONTAL_PADDING, OUTPUT_YOLOV8_CLASS_TEXT_HEIGHT);

        // Set the fill color for the text
        context.fillStyle = OUTPUT_YOLOV8_CLASS_TEXT_COLOR;

        // Draw the output class text within the bounding box
        context.fillText(outputClass, x1, y1 + 18);
    });
}

// Call the function to start the webcam stream
startWebcamStream();