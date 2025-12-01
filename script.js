let model;

// load model on startup
async function loadModel() {
    model = await tf.loadGraphModel("./model/model.json");
}
loadModel();

// receive POST from MIT App Inventor
addEventListener("message", async (event) => {
    const base64 = event.data;

    const image = new Image();
    image.src = "data:image/jpeg;base64," + base64;

    image.onload = async () => {
        const tensor = tf.browser.fromPixels(image)
            .resizeNearestNeighbor([224, 224])
            .expandDims(0)
            .toFloat()
            .div(tf.scalar(255));

        const prediction = await model.predict(tensor).data();

        const labels = ["bench", "dumbbell", "barbell", "rack", "cable_machine", "leg_press", "treadmill", "bike", "rower"];
        let results = [];

        prediction.forEach((p, i) => {
            if (p > 0.6) results.push(labels[i]);
        });

        // return result JSON
        fetch("/result", {
            method: "POST",
            body: JSON.stringify({ equipment_detected: results })
        });
    };
});
