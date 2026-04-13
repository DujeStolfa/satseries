//VERSION=3
function setup() {
    return {
        input: [{
            bands: ["SCL"],
            units: "DN"
        }],
        output: {
            bands: 1,
            sampleType: "INT16"
        }
    };
}

function evaluatePixel(sample) {
    return [sample.SCL];
}