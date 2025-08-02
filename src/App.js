import React, { useState, useRef } from "react";
import cv from "@techstark/opencv-js";
import * as ort from "onnxruntime-web"; // Changed to import * as ort for access to env
import Loader from "./components/loader";
import { detectImage } from "./utils/detect";
import { download } from "./utils/download";
import "./style/App.css";

const App = () => {
  const [session, setSession] = useState(null);
  const [loading, setLoading] = useState({ text: "Loading OpenCV.js", progress: null });
  const [image, setImage] = useState(null);
  const [inferenceTime, setInferenceTime] = useState(null);
  const uploadStartTime = useRef(null);
  const inputImage = useRef(null);
  const imageRef = useRef(null);
  const canvasRef = useRef(null);

  // Configs
  const modelName = "yolov8n.onnx";
  const modelInputShape = [1, 3, 640, 640];
  const topk = 100;
  const iouThreshold = 0.45;
  const scoreThreshold = 0.25;

  // wait until opencv.js initialized
  cv["onRuntimeInitialized"] = async () => {
    const baseModelURL = `${process.env.PUBLIC_URL}/model`;

    // Detect if on iOS or if SharedArrayBuffer (for multi-threading) is unavailable
    const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent) && !window.MSStream;
    if (isIOS || typeof SharedArrayBuffer === 'undefined') {
      ort.env.wasm.numThreads = 1; // Fallback to single-threaded WASM (CPU) for iPhone compatibility
      console.log('Fallback to single-threaded WASM enabled for better iOS support.');
    } else {
      // Optional: You can adjust threads based on device, but default is multi if available
      ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;
    }

    // create session with WebGL preferred, fallback to WASM
    const sessionOptions = { executionProviders: ['webgl', 'wasm'] };

    const arrBufNet = await download(
      `${baseModelURL}/${modelName}`, // url
      ["Loading YOLOv8 Segmentation model", setLoading] // logger
    );
    const yolov8 = await ort.InferenceSession.create(arrBufNet, sessionOptions);
    const arrBufNMS = await download(
      `${baseModelURL}/nms-yolov8.onnx`, // url
      ["Loading NMS model", setLoading] // logger
    );
    const nms = await ort.InferenceSession.create(arrBufNMS, sessionOptions);

    // warmup main model
    setLoading({ text: "Warming up model...", progress: null });
    const tensor = new ort.Tensor( // Updated to ort.Tensor
      "float32",
      new Float32Array(modelInputShape.reduce((a, b) => a * b)),
      modelInputShape
    );
    await yolov8.run({ images: tensor });

    setSession({ net: yolov8, nms: nms });
    setLoading(null);
  };

  return (
    <div className="App">
      {loading && (
        <Loader>
          {loading.text} {/* Removed progress display to avoid INFINITY% issue */}
        </Loader>
      )}
      <div className="header">
        <h1>YOLO Object Detection App</h1>
      
        <p>
          Serving : <code className="code">{modelName}</code>
        </p>
      </div>

      <div className="content">
        <img
          ref={imageRef}
          src="#"
          alt=""
          style={{ display: image ? "block" : "none" }}
          onLoad={async () => {
            const inferenceStartTime = Date.now();
            await detectImage(
              imageRef.current,
              canvasRef.current,
              session,
              topk,
              iouThreshold,
              scoreThreshold,
              modelInputShape
            );
            const inferenceEndTime = Date.now();
            
            const totalTime = inferenceEndTime - uploadStartTime.current;
            const inferenceOnlyTime = inferenceEndTime - inferenceStartTime;
            
            setInferenceTime({
              total: totalTime,
              inference: inferenceOnlyTime
            });
          }}
        />
        <canvas
          id="canvas"
          width={modelInputShape[2]}
          height={modelInputShape[3]}
          ref={canvasRef}
        />
        
      </div>
<div style={{ position: "absolute", bottom: "10px", left: "50%", transform: "translateX(-50%)", width: "100%", textAlign: "center" }}>
  {inferenceTime && (
          <div className="inference-time">
            <p>
              <strong>Total time (upload to detection):</strong> {inferenceTime.total}ms
            </p>
            <p>
              {/* <strong>Inference time:</strong> {inferenceTime.inference}ms */}
            </p>
          </div>
        )}
</div>
      <input
        type="file"
        ref={inputImage}
        accept="image/*"
        style={{ display: "none" }}
        onChange={(e) => {
          // handle next image to detect
          if (image) {
            URL.revokeObjectURL(image);
            setImage(null);
          }

          uploadStartTime.current = Date.now(); // capture upload start time
          setInferenceTime(null); // reset inference time
          
          const url = URL.createObjectURL(e.target.files[0]); // create image url
          imageRef.current.src = url; // set image source
          setImage(url);
        }}
      />
      <div className="btn-container">
        <button
          onClick={() => {
            inputImage.current.click();
          }}
        >
          Open local image
        </button>
        {image && (
          /* show close btn when there is image */
          <button
            onClick={() => {
              inputImage.current.value = "";
              imageRef.current.src = "#";
              URL.revokeObjectURL(image);
              setImage(null);
              setInferenceTime(null);
            }}
          >
            Close image
          </button>
        )}
      </div>
    </div>
  );
};

export default App;