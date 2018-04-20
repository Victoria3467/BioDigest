var CURRENT_URL = window.location.href.split('#')[0].split('?')[0],
    $BODY = $('body'),
    $MENU_TOGGLE = $('#menu_toggle'),
    $SIDEBAR_MENU = $('#sidebar-menu'),
    $SIDEBAR_FOOTER = $('.sidebar-footer'),
    $LEFT_COL = $('.left_col'),
    $RIGHT_COL = $('.right_col'),
    $NAV_MENU = $('.nav_menu'),
    $FOOTER = $('footer');

var dataUploadModal = null;
// Data upload modal
$(document).ready(function() {
  dataUploadModal = $('.data-upload-modal').modal({
    backdrop: 'static'
  });
});
// /Data upload modal

Dropzone.autoDiscover = false;


$(document).ready(function() {
  var myDropzone = new Dropzone("#data-upload", {
    maxFilesize: 10,
    createImageThumbnails: false,
    maxFiles: 1,
    acceptedFiles: ".csv"
  });

  myDropzone.on("success", function(file, response) {
    dataUploadModal.modal('hide');
    forecastChart.data.labels = response.forecast.xs;
    forecastChart.data.datasets[0].data = response.forecast.ys;
    forecastChart.update();

    inputsChart.data.labels = response.forecast.xs;
    for (var i = 0; i < response.data.length; i++) {
      inputData = response.data[i];
      inputsChart.data.datasets[i].data = inputData;
    }

    inputsChart.update();

    for (var i = 0; i < response.data.length; i++) {
      var newCols = '';
      for (var j = 0; j < response.data[i].length; j++) {
        newCols += '<td>' + response.data[i][j] + '</td>';
      }

      var newRow = '<tr class="' + (i % 2 == 0 ? 'even' : 'odd') + 'pointer"> \
                    <td>' + response.forecast.xs[i] + '</td>'
                    + newCols + '</tr>';

      $("#inputsTableBody").append(newRow);
    }

    var totals = response.totals;

    for (var i = 0; i < totals.length; i++) {
      var entry = totals[i];
      var newWidth = entry[1] / totals[0][1] * 100;
      $("#input-" + (i + 1) + "-header").text(entry[0]);
      $("#input-" + (i + 1) + "-progress").css('width', newWidth + '%');
    }

    var stats = ["entries", "revenue", "input", "accuracy"];

    for (var i = 0; i < stats.length; i++) {
      stat = stats[i];
      $("#stats-" + stat + "-header").text(response.stats[stat]);
      $("#stats-" + stat).text(response.stats[stat]);
    }

  });
});

var defaultForecastConfig = {
  type: 'line',
  data: {
    labels: ["January", "February", "March", "April", "May", "June", "July"],
    datasets: [{
    label: "Hainan BioCNG Production (m3)",
    backgroundColor: "rgba(38, 185, 154, 0.31)",
    borderColor: "rgba(38, 185, 154, 0.7)",
    pointBorderColor: "rgba(38, 185, 154, 0.7)",
    pointBackgroundColor: "rgba(38, 185, 154, 0.7)",
    pointHoverBackgroundColor: "#fff",
    pointHoverBorderColor: "rgba(220,220,220,1)",
    pointBorderWidth: 1,
    data: [31, 74, 6, 39, 20, 85, 7]
    }]
  }
};

var forecastChart = null;

var inputs = ["Pig Manure", "Cassava", "Fish Waste Water", "Kitchen Food", "Municipal Fecal Residue", "Tea", "Chicken Litter", "Bagasse Feed", "Alcohol", "Chinese Medicine", "Energy Grass", "Banana Fruit Shafts", "Lemon", "Percolate", "Other"];
var datasets = [];

var colors = {
  "Pig Manure": "rgba(239, 83, 80, 0.7)",
  "Cassava": "rgba(38, 198, 218, 0.7)",
  'Fish Waste Water': 'rgba(66, 165, 245, 0.7)',
  'Kitchen Food': 'rgba(38, 166, 154, 0.7)',
  'Municipal Fecal Residue': 'rgba(255, 202, 40, 0.7)',
  'Tea': 'rgba(126, 87, 194, 0.7)',
  'Chicken Litter': 'rgba(92, 107, 192, 0.7)',
  'Bagasse Feed': 'rgba(120, 144, 156, 0.7)',
  'Alcohol': 'rgba(141, 110, 99, 0.7)',
  'Chinese Medicine': 'rgba(236, 64, 122, 0.7)',
  'Energy Grass': 'rgba(255, 202, 40, 0.7)',
  'Banana Fruit Shafts': 'rgba(46, 125, 50, 0.7)',
  'Lemon': 'rgba(66, 66, 66, 0.7)',
  'Percolate': 'rgba(158, 157, 36, 0.7)',
  'Other': 'rgba(46, 125, 50, 0.7)'
}

var inputsData = inputs.map(function(inputType) {
  var data = [];
  for (var i = 0; i < 7; i++) {
    data.push(Math.floor((Math.random() * 100) + 1));
  }

  datasets.push({
    label: inputType,
    fill: false,
    borderColor: colors[inputType],
    pointBorderColor: colors[inputType],
    pointBackgroundColor: colors[inputType],
    pointHoverBackgroundColor: "#fff",
    pointHoverBorderColor: "rgba(220,220,220,1)",
    pointBorderWidth: 1,
    data: data
  });
});

var defaultInputsConfig = {
  type: 'line',
  data: {
    labels: ["January", "February", "March", "April", "May", "June", "July"],
    datasets: datasets
  }
};

var inputsChart = null;

$(document).ready(function() {
  var forecastCtx = document.getElementById("forecastChart");
  var inputsCtx = document.getElementById("inputsChart");
  forecastChart = new Chart(forecastCtx, defaultForecastConfig);
  inputsChart = new Chart(inputsCtx, defaultInputsConfig);
});
