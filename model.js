// ===============
// Load the model
// ===============
let modelLoaded = false;
let model;

async function loadModel() {
    model = await tf.loadGraphModel("model/model.json");
    modelLoaded = true;
    console.log("Model loaded");
}
loadModel();


// ============================
// Handle POST from MIT Inventor
// ============================
addEventListener("fetch", event => {
    event.respondWith(handleRequest(event.request));
});


async function handleRequest(request) {
    // Only accept POST
    if (request.method !== "POST") {
        return new Response("Send Base64 image using POST", { status: 400 });
    }

    // Get Base64 text
    const base64 = await request.text();

    if (!modelLoaded) {
        return new Response(JSON.stringify({
            error: "Model not loaded"
        }), { status: 503 });
    }

    // Convert Base64 → Image
    const img = await loadImage("data:image/jpeg;base64," + base64);

    // Convert Image → Tensor
    const inputTensor = tf.browser.fromPixels(img)
        .resizeNearestNeighbor([224, 224])
        .expandDims(0)
        .toFloat()
        .div(tf.scalar(255));

    // Run prediction
    const predictionOutput = await model.predict(inputTensor).data();

    // Map model output to equipment labels
    const labels = [
        "bench_press",
        "dumbbells",
        "barbell",
        "squat_rack",
        "lat_pulldown",
        "leg_press",
        "treadmill",
        "rowing_machine",
        "exercise_bike",
        "cable_machine"
    ];

    let found = [];

    for (let i = 0; i < predictionOutput.length; i++) {
        if (predictionOutput[i] > 0.50) {         // Confidence threshold
            found.push(labels[i]);
        }
    }

    return new Response(JSON.stringify({
        equipment_detected: found
    }), {
        status: 200,
        headers: { "Content-Type": "application/json" }
    });
}


// ==============
// Helper function
// ==============
function loadImage(url) {
    return new Promise(resolve => {
        const img = new Image();
        img.crossOrigin = "anonymous";
        img.src = url;
        img.onload = () => resolve(img);
    });
}
