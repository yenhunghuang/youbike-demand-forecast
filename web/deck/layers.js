const MAPBOX_TOKEN = "pk.test-deckgl-placeholder-token";

const initialViewState = {
  longitude: 121.544,
  latitude: 25.034,
  zoom: 12,
  pitch: 45,
  bearing: 0,
};

const sampleStations = [
  { name: "Taipei City Hall", coordinates: [121.5645, 25.0375], residual: -1.2 },
  { name: "Sun Yat-sen Memorial", coordinates: [121.5602, 25.0404], residual: 0.8 },
  { name: "Songshan Station", coordinates: [121.577, 25.048], residual: 1.5 },
];

const horizonResiduals = {
  t15: sampleStations,
  t30: sampleStations.map((s) => ({ ...s, residual: s.residual * 1.1 })),
  t60: sampleStations.map((s) => ({ ...s, residual: s.residual * 1.3 })),
};

function createScatterLayer(horizonKey) {
  return new deck.ScatterplotLayer({
    id: `residual-scatter-${horizonKey}`,
    data: horizonResiduals[horizonKey],
    getPosition: (d) => d.coordinates,
    getRadius: (d) => Math.abs(d.residual) * 500,
    getFillColor: (d) => (d.residual > 0 ? [255, 99, 71, 200] : [0, 204, 153, 200]),
    pickable: true,
  });
}

function createHeatmapLayer(horizonKey) {
  return new deck.HeatmapLayer({
    id: `residual-heatmap-${horizonKey}`,
    data: horizonResiduals[horizonKey],
    getPosition: (d) => d.coordinates,
    getWeight: (d) => d.residual,
    radiusPixels: 60,
  });
}

const deckgl = new deck.DeckGL({
  container: "map",
  initialViewState,
  controller: true,
  getTooltip: ({ object }) =>
    object && {
      html: `<strong>${object.name}</strong><br/>Residual: ${object.residual.toFixed(2)}`,
    },
  mapStyle: "mapbox://styles/mapbox/dark-v11",
  mapboxAccessToken: MAPBOX_TOKEN,
});

const horizonSelect = document.getElementById("horizon-select");
const summary = document.getElementById("layer-summary");

function updateLayers() {
  const horizonKey = horizonSelect.value;
  const scatterLayer = createScatterLayer(horizonKey);
  const heatmapLayer = createHeatmapLayer(horizonKey);
  deckgl.setProps({ layers: [scatterLayer, heatmapLayer] });
  summary.textContent = `Rendering ${horizonKey.toUpperCase()} residual layers (scatter + heatmap) for ${scatterLayer.props.data.length} stations.`;
}

horizonSelect.addEventListener("change", updateLayers);
updateLayers();
