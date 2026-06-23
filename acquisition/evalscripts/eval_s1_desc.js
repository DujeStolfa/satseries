//VERSION=3
function setup() {
    return {
        input: ["VV", "VH", "dataMask"],
        output: {
            id: "default",
            bands: 3,
            sampleType: "FLOAT32"
        }
    };
}

function evaluatePixel(sample) {
    return [
        10 * Math.log(sample.VV) / Math.LN10,
        10 * Math.log(sample.VH) / Math.LN10,
        sample.dataMask
    ];
}