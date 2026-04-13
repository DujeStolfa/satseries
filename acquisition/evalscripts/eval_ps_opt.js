//VERSION=3
function setup() {
    return {
        input: [{
            "bands": ["coastal_blue", "blue", "green_i", "green", "yellow", "red", "rededge", "nir"]
        }],
        output: {
            bands: 8,
            sampleType: "INT16"
        }
    }
}

function evaluatePixel(sample) {
    return [
        sample.coastal_blue,
        sample.blue,
        sample.green_i,
        sample.green,
        sample.yellow,
        sample.red,
        sample.rededge,
        sample.nir
    ]
}