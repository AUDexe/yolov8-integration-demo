importScripts("https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js");

const YOLOV8_ONNX_FILE_PATH = './yolov8m.onnx';

const YOLOV8_INPUT_IMAGE_COLOR_CHANNELS = 3;
const YOLOV8_INPUT_IMAGE_WIDTH = 640;
const YOLOV8_INPUT_IMAGE_HEIGHT = 640;

onmessage = async (event) => {
    const modelInput = event.data;
    const modelOutput = await runYolov8Model(modelInput);
    postMessage(modelOutput);
}

async function runYolov8Model(input) {
    // Load the model from the ONNX file
    const model = await ort.InferenceSession.create(YOLOV8_ONNX_FILE_PATH);

    // Convert the input data to a Float32Array and reshape it to match the expected input shape of the model
    input = new ort.Tensor(Float32Array.from(input), [1, YOLOV8_INPUT_IMAGE_COLOR_CHANNELS, YOLOV8_INPUT_IMAGE_WIDTH, YOLOV8_INPUT_IMAGE_HEIGHT]);

    // Run the model with constructed tensor and receive outputs
    const outputs = await model.run({ images: input });

    // Return the data of the first output
    return outputs['output0'].data;
}