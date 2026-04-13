//VERSION=3
function setup() {
    return {
        input: [{
            "bands": ["clear", "snow", "shadow", "haze_light", "haze_heavy", "cloud", "confidence", "udm1"]
        }],
        output: {
            bands: 8,
            sampleType: "UINT8"
        }
    }
}

function evaluatePixel(sample) {
    return [
        sample.clear,
        sample.snow,
        sample.shadow,
        sample.haze_light,
        sample.haze_heavy,
        sample.cloud,
        sample.confidence,
        sample.udm1
    ]
}