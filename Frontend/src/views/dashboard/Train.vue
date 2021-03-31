<template>
  <v-container
    id="dashboard"
    fluid
    tag="section"
  >
  <v-alert
    border="top"
    colored-border
    type="info"
    elevation="2"
  >
    Apply training strategies to improve performance of existing models
  </v-alert>
  <br/>
    <div v-if="vis">
    <center>
      <v-col
        cols="12"
        md="6"
      >
        <v-text-field
          v-model="name"
          label="Name of new model"
        ></v-text-field>
      </v-col>
    </center>
    <h3>Select Method:</h3><br/>
    <v-select
      v-model="method"
      :items="['Train further', 'Incremental Learning']"
      label="Select Dataset"
      solo
    ></v-select>
    <h3>Select Options: </h3><br/>
    <v-row>
      <v-col
        cols="12"
        md="4"
      >
        <v-subheader class="pl-0">
          Dataset to use
        </v-subheader>
        <v-select
          v-model="data"
          :items="items1"
          label="Choose an option"
        ></v-select>
      </v-col>
      <v-col
        cols="12"
        md="4"
      >
        <v-subheader v-if="method == 'Train further'" class="pl-0">
          Preload Weights from
        </v-subheader>
        <v-subheader v-else class="pl-0">
          Model to improve
        </v-subheader>
        <v-select
          v-model="preload"
          :items="items"
          label="Choose a model"
        ></v-select>
      </v-col>
      <v-col
        cols="12"
        md="4"
        v-if="method == 'Train further'"
      >
        <v-subheader class="pl-0">
          Early stopping criteria
        </v-subheader>
        <v-select
          v-model="early"
          :items="items2"
          label="Choose an option"
        ></v-select>
      </v-col>
      <v-col
        cols="12"
        md="4"
      >
      <v-subheader class="pl-0">
        Epochs
      </v-subheader>
      <v-slider
        v-if="method == 'Incremental Learning'"
        v-model="epochs"
        :thumb-size="24"
        thumb-label
        thumb-label="always"
        step="25"
        ticks
      >
        <template v-slot:thumb-label="{ value }">
          {{ inc_choices[Math.min(Math.floor(value / 25), 4)] }}
        </template>
      </v-slider>
      <v-slider
        v-else
        v-model="epochs"
        :thumb-size="24"
        thumb-label
        thumb-label="always"
      ></v-slider>
      </v-col>
      <v-col
        cols="12"
        md="4"
      >
      <v-subheader class="pl-0">
        Batch Size
      </v-subheader>
      <v-slider
        v-model="batch_size"
        :thumb-size="24"
        thumb-label
        thumb-label="always"
      >
      </v-slider>
      </v-col>
      <v-col
        cols="12"
        md="4"
      >
      <v-checkbox
        v-model="aug"
        label="Apply transformations while training"
      ></v-checkbox>
      </v-col>
      <v-col
        v-if="preload && preload != 'None' && method == 'Train further'"
        cols="12"
        md="4"
      >
      <v-checkbox
        v-model="freeze"
        label="Freeze loaded weights"
      ></v-checkbox>
      </v-col>
      <v-col
        v-if="early == 'Threshold validation accuracy' && method == 'Train further'"
        cols="12"
        md="4"
      >
      <v-subheader class="pl-0">
        Threshold Validation Accuracy for stop
      </v-subheader>
      <v-slider
        v-model="val_acc"
        :thumb-size="24"
        thumb-label
        thumb-label="always"
      >
      </v-slider>
      </v-col>
      <v-col
        v-if="early == 'Decrease in performance observed' && method == 'Train further'"
        cols="12"
        md="4"
      >
      <v-subheader class="pl-0">
        Threshold epochs (if performance decreasing)
      </v-subheader>
      <v-slider
        v-model="thres_epoch"
        :thumb-size="24"
        thumb-label
        thumb-label="always"
      >
      </v-slider>
      </v-col>
    </v-row>
    <h3>Select Optimizer & Scheduler Options: </h3><br/>
    <v-row>
      <v-col
        cols="12"
        md="3"
      >
        <v-text-field
          v-model="lr"
          label="Learning Rate"
          type="number"
        ></v-text-field>
      </v-col>
      <v-col
        cols="12"
        md="3"
        v-if="method == 'Train further'"
      >
        <v-text-field
          v-model="momentum"
          label="Momentum"
          type="number"
        ></v-text-field>
      </v-col>
      <v-col
        cols="12"
        md="3"
      >
        <v-text-field
          v-model="decay"
          label="Gamma"
          type="number"
        ></v-text-field>
      </v-col>
      <v-col
        cols="12"
        md="3"
        v-if="method == 'Train further'"
      >
        <v-text-field
          v-model="l2_norm"
          label="L2 Normalization"
          type="number"
        ></v-text-field>
      </v-col>
    </v-row>
    <br/><br/>
    <center>
      <v-btn
        color="primary"
        @click="start()"
      >
        Start Training
      </v-btn>
    </center>
    </div>
    <div v-else>
      <v-alert
        :value="current.saved"
        color="primary"
        dark
        border="top"
        type="success"
        transition="scale-transition"
      >
        Model saved. Better performing models will continue to overwrite the saved model.
      </v-alert>
      <center>
        <h2 v-if="method == 'Train further'">Training Report (Updates every 10 secs)</h2>
        <h2 v-else>Training Report (Updates every sec)</h2>
      </center>
      <br/><br/>
      <v-row align="center">
        <v-col
          cols="12"
          md="4"
        >
        <center>
          <apexchart type="radialBar" height="350" :options="chartOptions" :series="series"></apexchart>
        </center>
        </v-col>
        <v-col
          cols="12"
          md="4"
        >
        <center v-if="current.running">
          <v-progress-circular
            indeterminate
            color="primary"
          ></v-progress-circular>
        </center>
        <br/><br/>
        <v-simple-table>
          <template v-slot:default>
            <tbody>
              <tr>
                <td><strong>Current Epoch:</strong></td>
                <td class="text-right">{{current.train_epoch}}</td>
              </tr>
              <tr>
                <td><strong>Current Running Loss:</strong></td>
                <td class="text-right">{{current.train_loss.toFixed(3)}}</td>
              </tr>
              <tr>
                <td><strong>Last Training Loss:</strong></td>
                <td class="text-right" v-if="current.avg_train_loss.length">{{current.avg_train_loss[current.avg_train_loss.length-1].toFixed(3)}}</td>
                <td class="text-right" v-else>NA</td>
              </tr>
              <tr>
                <td><strong>Last Validation Loss:</strong></td>
                <td class="text-right" v-if="current.val_loss.length">{{current.val_loss[current.val_loss.length-1].toFixed(3)}}</td>
                <td class="text-right" v-else>NA</td>
              </tr>
            </tbody>
          </template>
        </v-simple-table>
        </v-col>
        <v-col
          cols="12"
          md="4"
        >
          <apexchart type="radialBar" ref="accu" height="350" :options="accChartOptions" :series="acc"></apexchart>
        </v-col>
      </v-row>
      <center>
        <v-col
          cols="12"
          md="6"
        >
        <apexchart type="line" height="500" ref="chart" :options="losschartOptions" :series="losses"></apexchart>
        </v-col>
        <br/>
        <v-btn
          v-if="current.running"
          color="primary"
          @click="stop()"
        >
          Stop Training
        </v-btn>
        <v-btn
          v-else
          color="primary"
          @click="$router.push('evaluate')"
        >
          Evaluate Model
        </v-btn>
      </center>
    </div>
  </v-container>
</template>

<style>

</style>

<script>
  import axios from 'axios';

  export default {
    name: 'Train',
    components: {

    },
    data () {
      return {
        method: 'Train further',
        inc: false,
        inc_choices: ['1', '2', '3', '4', '5'],
        prev_classes: null,
        aug: true,
        freeze: false,
        losses: [
          {
            name: "Validation Loss",
            data: []
          },
          {
            name: "Training Loss",
            data: []
          }
        ],
        vis: true,
        updatedlabel: false,
        data: null,
        preload: null,
        current: {
          "running": true,
          "train_loss": 0,
          "avg_train_loss": [],
          "val_loss": [],
          "train_epoch": 0,
          "val_accuracy": 0,
          "saved": false
        },
        name: null,
        early: null,
        val_acc: 95,
        thres_epoch: 10,
        items: ["None", "Benchmark model"],
        items1: ["Main Dataset", "GTSRB_48 Dataset", "Difficult Dataset"],
        items2: ["None", "Decrease in performance observed", "Threshold validation accuracy"],
        epochs: 50,
        batch_size: 50,
        lr: 0.007,
        momentum: 0.8,
        decay: 0.9,
        l2_norm: 0.00001,
        series: [0],
        chartOptions: {
          chart: {
            height: 350,
            type: 'radialBar',
          },
          plotOptions: {
            radialBar: {
              hollow: {
                size: '70%',
              }
            },
          },
          labels: ['Progress'],
        },
        acc: [0],
        accChartOptions: {
          chart: {
            height: 350,
            type: 'radialBar',
            offsetY: -10
          },
          plotOptions: {
            radialBar: {
              startAngle: -135,
              endAngle: 135,
              dataLabels: {
                name: {
                  fontSize: '16px',
                  color: undefined,
                  offsetY: 120
                },
                value: {
                  offsetY: 76,
                  fontSize: '22px',
                  color: undefined,
                  formatter: function (val) {
                    return val + "%";
                  }
                }
              }
            }
          },
          fill: {
            type: 'gradient',
            gradient: {
                shade: 'dark',
                shadeIntensity: 0.15,
                inverseColors: false,
                opacityFrom: 1,
                opacityTo: 1,
                stops: [0, 50, 65, 91]
            },
          },
          stroke: {
            dashArray: 4
          },
          labels: ['Validation Accuracy'],
        },
        losschartOptions: {
            chart: {
              height: 350,
              type: 'line',
              zoom: {
                enabled: false
              }
            },
            dataLabels: {
              enabled: false
            },
            stroke: {
              curve: 'straight'
            },
            title: {
              text: 'Loss v/s Epoch',
              align: 'left'
            },
            grid: {
              row: {
                colors: ['#f3f3f3', 'transparent'], // takes an array which will be repeated on columns
                opacity: 0.5
              },
            },
            xaxis: {
              min: 0,
              max: 100,
              title: {
                text: 'Epoch'
              },
              type: "numeric",
              labels: {
                formatter: function (value) {
                  return value.toFixed(0);
                }
              },
            },
            yaxis: {
              min: 0,
              title: {
                text: 'Loss'
              },
            }
          },
      }
    },
    methods: {
      start(){
        var _this = this;
        var num = null;
        if((!this.early && this.method=="Train further") || (!this.preload && this.method=="Train further") || !this.data){
          _this.$notify({title: 'Error', type: 'error', text: "Please select an option!"})
        }else if(!this.name){
          _this.$notify({title: 'Error', type: 'error', text: "Please enter the name of new model!"})
        }else{
          var data = null;
          if(this.data == "Main Dataset"){
            data = "main"
            num = this.$store.state.num_classes
          }else if(this.data == "Difficult Dataset"){
            data = "diff"
            num = 48
          }else if(this.data == "GTSRB_48 Dataset"){
            data = "base"
            num = 48
          }else{
            num = 43
          }

          var epoch = this.epochs;
          if(this.method == "Incremental Learning"){
            this.inc = true;
            epoch /= 25;
            epoch++;
          }else{
            this.inc = false;
          }

          var val_acc = this.val_acc;
          var thres_epoch = this.thres_epoch;
          if(this.early == "Decrease in performance observed"){
            val_acc = null
          }else if(this.early == "Threshold validation accuracy"){
            thres_epoch = null
          }else{
            val_acc = null;
            thres_epoch = null
          }
          axios.post(_this.$store.state.server + '/trainmodel', {
              inc: _this.inc,
              prev_classes: _this.prev_classes,
              prev_name: _this.preload,
              num_classes: num,
              batch_size: _this.batch_size,
              epochs: epoch,
              lr: Number(_this.lr),
              momentum: Number(_this.momentum),
              decay: Number(_this.decay),
              step: 1000,
              l2_norm: Number(_this.l2_norm),
              name: _this.name,
              thres_acc: val_acc,
              thres_epoch: thres_epoch,
              data: data,
              freeze: _this.freeze,
              aug: _this.aug
          }).then(function (response){
              _this.$notify({title: 'Successful', type: 'success', text: response.data})
              _this.losschartOptions.xaxis.max = epoch;
              _this.vis = false;
              _this.epochs = epoch;
              _this.val_acc = val_acc;
              _this.thres_epoch = thres_epoch;
              _this.refresh();
          }).catch(function (error){
              _this.$notify({title: 'Error', type: 'error', text: error.message})
          });
        }
      },
      refresh(){
        var _this = this
        var interval = 10000;
        if(this.method == "Incremental Learning"){
          interval = 1000;
        }
        var x = setInterval(function(){
          if(!_this.updatedlabel && _this.inc){
            _this.updatedlabel = true;
            _this.$refs.accu.updateOptions({
              labels: ['Validation Accuracy (Extra Classes)']
            })
          }
          axios.get(_this.$store.state.server + '/traininfo').then(function (response){
              if(response.data.val_loss && response.data.val_loss.length!=_this.current.val_loss.length){
                var loss0 = []
                var loss1 = []
                for(var i=0; i<response.data.val_loss.length; i++){
                  loss0.push({x: i+1, y: (response.data.val_loss[i]).toFixed(3)})
                  loss1.push({x: i+1, y: (response.data.avg_train_loss[i]).toFixed(3)})
                }
                _this.$refs.chart.updateSeries([
                  {
                    name: "Validation Loss",
                    data: loss0
                  }, {
                    name: "Training Loss",
                    data: loss1
                  }
                ]);
              }
              _this.current = response.data;
              if(_this.current.train_epoch){
                _this.series = [((_this.current.train_epoch-1) / _this.epochs * 100).toFixed(2)];
              }
              _this.acc = [(_this.current.val_accuracy).toFixed(2)];
              if(!_this.current.running){
                _this.series = [100]
                clearInterval(x);
              }
          })
        }, interval);
      },
      stop(){
        var _this = this
        axios.get(_this.$store.state.server + '/stoptraining').then(function (response){
            _this.$notify({title: 'Successful', type: 'success', text: response.data})
        }).catch(function (error){
            _this.$notify({title: 'Error', type: 'error', text: error.message})
        });
      }
    },
    mounted(){
      var _this = this;
      axios.get(_this.$store.state.server + '/modelinfo').then(function (response){
          for(var x in response.data.result){
            _this.items.push(response.data.result[x].slice(0,-4))
          }
      })
    },
    watch: {
      preload: function(){
        if(this.preload == "None"){
          this.prev_classes = null
        }else if(this.preload == "Benchmark model"){
          this.prev_classes = 43
        }else{
          var _this = this;
          axios.post(_this.$store.state.server + '/modelstats', {
              name: _this.preload
          }).then(function (response){
              _this.prev_classes = response.data.num_classes;
          })
        }
      }
    }
  }
</script>
