$(document).ready(function() {
  var mymap = L.map('map', {
    center: [32, 100],
    zoom: 4,
    maxBounds: null,
    layers: [],
    worldCopyJump: false,
    crs: L.CRS.EPSG3857
  });

  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    "attribution": null,
    "detectRetina": false,
    "maxZoom": 18,
    "minZoom": 1,
    "noWrap": false,
    "subdomains": "abc"
  }).addTo(mymap);
});
