var dataUploadModal,
    forecastChart,
    inputsChart;


var forecastCtx = $("#forecastChart"),
    inputsCtx = $("#inputsChart");

var INPUTS = [
  'Kitchen Food',
  'Kitchen Food Paste',
  'Tea',
  'Fruit and Vegetables',
  'Bread Paste',
  'Flour and Waste Oil',
  'Oil',
  'Lemon',
  'Cassava',
  'Alcohol',
  'Banana Fruit Shafts',
  'Chinese Medicine',
  'Energy Grass',
  'Pig Manure',
  'Fish Waste Water',
  'Chicken Litter',
  'Bagasse Feed',
  'Municipial Fecal Residue',
  'Diesel Waste Water',
  'Percolate',
  'Acid Feed',
  'Acid Discharge',
  'Other'
]

var SAMPLE_MONTHS = ["January", "February", "March", "April", "May", "June", "July"];

function randArray(max, len) {
  var arr = [];
  for (var i = 0; i < len; i++) {
    arr.push(_.random(0, max));
  }

  return arr;
}

function randColor(alpha) {
  r = _.random(0, 256), g = _.random(0, 256), b = _.random(0, 256);
  return 'rgba(' + r + ', ' + g + ', ' + b + ', ' + alpha + ')';
}

function setUpModal() {
  Dropzone.autoDiscover = false;

  dataUploadModal = $('.data-upload-modal').modal({
    // backdrop: 'static'
  });

  var myDropzone = new Dropzone("#data-upload", {
    maxFilesize: 10,
    createImageThumbnails: false,
    maxFiles: 1,
    acceptedFiles: ".csv"
  });

  myDropzone.on("success", function(file, response) {
    dataUploadModal.modal('hide');

    // Update forecasts chart
    forecastChart.data.labels = response.forecast.xs;
    forecastChart.data.datasets[0].data = response.forecast.ys;
    forecastChart.data.datasets[1].data = response.shenzhen_predictions;
    forecastChart.update();

    // Update inputs chart
    inputsChart.data.labels = response.forecast.xs;

    for (var i = 0; i < 10; i++) {
      inputsChart.data.datasets[i].data = response.data_top[i];
      inputsChart.data.datasets[i].label = response.data_columns[i];
    }

    inputsChart.update();

    // Update inputs table
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

    // Update inputs contributions graph
    var totals = response.totals;
    for (var i = 0; i < totals.length; i++) {
      var entry = totals[i];
      var newWidth = entry[1] / totals[0][1] * 100;
      $("#input-" + (i + 1) + "-header").text(entry[0]);
      $("#input-" + (i + 1) + "-progress").css('width', newWidth + '%');
    }

    // Update statistics cards
    var stats = ["entries", "revenue", "input", "accuracy"];
    for (var i = 0; i < stats.length; i++) {
      stat = stats[i];
      $("#stats-" + stat + "-header").text(response.stats[stat]);
      $("#stats-" + stat).text(response.stats[stat]);
    }
  });
}

function setUpGraphs() {
  var defaultForecastConfig = {
    type: 'line',
    data: {
      labels: SAMPLE_MONTHS,
      datasets: [{
        label: "Hainan Revenue Forecast (USD)",
        backgroundColor: "rgba(38, 185, 154, 0.31)",
        borderColor: "rgba(38, 185, 154, 0.7)",
        pointBorderColor: "rgba(38, 185, 154, 0.7)",
        pointBackgroundColor: "rgba(38, 185, 154, 0.7)",
        pointHoverBackgroundColor: "#fff",
        pointHoverBorderColor: "rgba(220,220,220,1)",
        pointBorderWidth: 1,
        data: randArray(100, 7)
      }, {
        label: "Shenzhen Revenue Forecast (USD)",
        backgroundColor: "rgba(233, 30, 99, 0.31)",
        borderColor: "rgba(233, 30, 99, 0.7)",
        pointBorderColor: "rgba(233, 30, 99, 0.7)",
        pointBackgroundColor: "rgba(233, 30, 99, 0.7)",
        pointHoverBackgroundColor: "#fff",
        pointHoverBorderColor: "rgba(220,220,220,1)",
        pointBorderWidth: 1,
        data: randArray(100, 7)
      }]
    }
  };

  var inputsDatasets = [];

  var inputsData = _.map(INPUTS.slice(0, 10), function(inputType) {
    var data = randArray(100, 7);
    var color = randColor(1);

    inputsDatasets.push({
      label: inputType,
      fill: false,
      borderColor: color,
      pointBorderColor: color,
      pointBackgroundColor: color,
      pointHoverBackgroundColor: "#fff",
      pointHoverBorderColor: "rgba(220,220,220,1)",
      pointBorderWidth: 1,
      data: data
    });
  });

  var defaultInputsConfig = {
    type: 'line',
    data: {
      labels: SAMPLE_MONTHS,
      datasets: inputsDatasets
    }
  };

  forecastChart = new Chart(forecastCtx, defaultForecastConfig);
  inputsChart = new Chart(inputsCtx, defaultInputsConfig);
}


$(document).ready(function() {
  setUpModal();
  setUpGraphs();
});
