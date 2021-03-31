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
    Generate evaluation metrics for available datasets on existing models
  </v-alert>
  <br/>
  <div v-if="vis">
    <div v-if="num_queue">
      <h1>Evaluations:</h1>
      <br/>
      <div v-for="(x, idx) in num_queue" :key="idx">
        <hr/>
        <br/>
        <h3>Model: {{report_data[idx].model}}</h3>
        <h3>Dataset: {{report_data[idx].dataset}}</h3>
        <br/>
        <v-progress-linear
          color="light-blue"
          height="10"
          :value="$store.state.eval_prog[idx]"
          striped
        ></v-progress-linear>
        <br/>
        <h3>{{$store.state.eval_prog[idx]}}%</h3>
        <center v-if="$store.state.eval_prog[idx] == 100 && !running[idx]">
          <v-btn
            color="primary"
            small
            @click="vis = false; current = idx"
          >
            View Report
          </v-btn>
          <br/>
        </center>
        <br/>
        <hr/>
      </div>
    </div>
    <br/><br/>
    <center>
      <v-col
        class="d-flex"
        cols="12"
        sm="8"
      >
        <v-select
          :items="items"
          v-model="choice"
          label="Dataset to use"
          dense
          outlined
        ></v-select>
        <v-select
          :items="items2"
          v-model="choice1"
          label="Classes to evaluate"
          dense
          outlined
        ></v-select>
        <v-select
          :items="items1"
          v-model="model"
          label="Model to evaluate"
          dense
          outlined
        ></v-select>
      </v-col>

      <v-col
        v-if="model"
        cols="12"
        md="6"
      >
      <h3>Selected model stats:</h3><br/>
      <v-simple-table>
        <template v-slot:default>
          <tbody>
            <tr>
              <td><strong>Number of classes model trained on:</strong></td>
              <td class="text-right">{{model_param.num_classes}}</td>
            </tr>
            <tr>
              <td><strong>Validation Accuracy:</strong></td>
              <td class="text-right">{{model_param.val_acc.toFixed(2)}}%</td>
            </tr>
            <tr>
              <td><strong>Last Training Epochs:</strong></td>
              <td class="text-right">{{model_param.epochs}}</td>
            </tr>
            <tr>
              <td><strong>Last Training Loss:</strong></td>
              <td class="text-right">{{model_param.train_loss.toFixed(3)}}</td>
            </tr>
            <tr>
              <td><strong>Last Validation Loss:</strong></td>
              <td class="text-right">{{model_param.val_loss.toFixed(3)}}</td>
            </tr>
          </tbody>
        </template>
      </v-simple-table>
      </v-col>

      <br/><br/>
      <v-btn
        color="primary"
        @click="evaluate(num_queue)"
      >
        Evaluate
      </v-btn>
    </center>
    </div>
    <div v-else>
      <center>
        <v-btn
          color="primary"
          @click="vis = true"
        >
          Close Report
        </v-btn>
      </center>
      <br/>
      <Report :eval_data="report_data[current]"/>
      <center>
        <v-btn
          style="margin-right:0px"
          color="primary"
          @click="vis = true"
        >
          Close Report
        </v-btn>
      </center>
    </div>
  </v-container>
</template>

<style>

</style>

<script>
import axios from 'axios';
import Report from '@/views/dashboard/Report';

  export default {
    name: 'Evaluate',
    components: {
      Report
    },
    data () {
      return {
        num_queue: 0,
        current: 0,
        running: [false, false, false, false, false],
        report_data: [{}, {}, {}, {}, {}],
        model_param: {
          "method": "Train further",
          "num_classes": 43,
          "epochs": 193,
          "val_acc": 96.00,
          "val_loss": 0.396,
          "train_loss": 1.747,
          "l2_norm": 0.00001,
          "lr": 0.007,
          "momentum": 0.8,
          "gamma": 0.9
        },
        macro: {
          "f1": 0,
          "prec": 0,
          "reca": 0,
          "spec": 0,
          "posl": 0,
          "negl": 0,
          "bcrt": 0,
          "bert": 0,
          "matc": 0,
        },
        num_classes: 43,
        sel_image: null,
        sel_class: null,
        page: [],
        dialog: false,
        heat: null,
        acc: [],
        tabledata: [],
        misclas: [],
        results: [null, null, null, null, null],
        vis: true,
        choice: null,
        choice1: null,
        model: null,
        items: ["Main dataset", "GTSRB dataset", "GTSRB_48 dataset", "Difficult dataset"],
        items1: ['Benchmark Model'],
        items2: ['Original 43 classes', 'All classes'],
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
          labels: ['Accuracy'],
        },
      }
    },
    methods: {
      getModelInfo(){
        var _this = this;
        var name = _this.model;
        axios.post(_this.$store.state.server + '/modelstats', {
            name: name
        }).then(function (response){
            _this.model_param = response.data;
        }).catch(function (error){
            _this.$notify({title: 'Error', type: 'error', text: error.message})
        });
      },
      getResult(i){
        var _this = this;
        axios.post(_this.$store.state.server + '/eval_result', {
            job_num: i
        }).then(function (response){
            _this.results[i] = response.data.result;
            _this.stats(i);
            _this.$set(_this.running, i, false);
        }).catch(function (error){
            _this.$notify({title: 'Error', type: 'error', text: error.message})
        });
      },
      evaluate(i){
        var _this = this;

        if(!this.choice || !this.model || !this.choice1){
          _this.$notify({title: 'Error', type: 'error', text: 'Please select an option'})
        }else if(this.num_queue == 5){
          _this.$notify({title: 'Error', type: 'error', text: 'Maximum job limit reached! Reload to continue'})
        }else{
          if(this.choice == "GTSRB dataset" && this.choice1 == "All classes"){
            _this.$notify({title: 'Error', type: 'error', text: 'Original dataset has only 43 classes!'})
          }else{
            this.running[i] = true;
            var dataset = null;

            if(this.choice == "Main dataset"){
              dataset = "main"
            }else if(this.choice == "GTSRB dataset"){
              dataset = "orig"
            }else if(this.choice == "GTSRB_48 dataset"){
              dataset = "base"
            }else{
              dataset = "diff"
            }

            if(this.choice1 == "All classes"){
              this.num_classes = this.$store.state.num_classes;
            }else{
              this.num_classes = 43;
            }

            var name = _this.model;
            if(name == "Benchmark Model"){
              name = null;
            }
            if(_this.model_param.num_classes <= _this.num_classes){
              axios.post(_this.$store.state.server + '/evaluate', {
                  nclasses: _this.num_classes,
                  dataset: dataset,
                  name: name,
                  job_num: _this.num_queue
              }).then(function (response){
                  _this.$notify({title: 'Successful', type: 'success', text: response.data})
                  _this.report_data[i].model_param = _this.model_param;
                  _this.report_data[i].num_classes = _this.num_classes;
                  _this.report_data[i].dataset = _this.choice;
                  _this.report_data[i].model = _this.model;
                  _this.num_queue++;
              }).catch(function (error){
                  _this.$notify({title: 'Error', type: 'error', text: error.message})
                  _this.running[i] = false;
              });
            }else{
              _this.$notify({title: 'Error', type: 'error', text: 'Cannot evaluate on less no. of classes than no. of classes trained on!'})
              _this.running[i] = false;
            }
          }
        }
        var x = setInterval(function(){
          axios.get(_this.$store.state.server + '/evalprogress').then(function (response){
            _this.$store.commit('updateEvalProg', response.data.prog);
            if(response.data.prog[i] >= 100 && _this.running[i] == true){
              _this.getResult(i);
              clearInterval(x);
            }
          })
        }, 5000);
      },
      stats(idx){
        var macro = {};
        const average = arr => arr.reduce( ( p, c ) => p + c, 0 ) / arr.length;

        this.tabledata = [];
        this.acc = [];
        this.misclas = [];
        for(var i=0; i<this.report_data[idx].num_classes; i++){
          this.misclas.push([]);
        }
        var conf = new Array(this.report_data[idx].num_classes).fill().map(() => Array(this.report_data[idx].num_classes).fill(0));
        var pres = new Array(this.report_data[idx].num_classes).fill(0);
        var reca = new Array(this.report_data[idx].num_classes).fill(0);
        var spec = new Array(this.report_data[idx].num_classes).fill(0);
        var posl = new Array(this.report_data[idx].num_classes).fill(0);
        var negl = new Array(this.report_data[idx].num_classes).fill(0);
        var bcrt = new Array(this.report_data[idx].num_classes).fill(0);
        var bert = new Array(this.report_data[idx].num_classes).fill(0);
        var matc = new Array(this.report_data[idx].num_classes).fill(0);
        var f1 = new Array(this.report_data[idx].num_classes).fill(0);
        var corr = 0;
        var wrng = 0;

        for(var x=0; x<this.results[idx].length; x++){
          var p = this.results[idx][x]['Pred'];
          var a = this.results[idx][x]['Actual'];
          if(p == a){
            corr++;
            conf[p][p]++;
          }else{
            wrng++;
            conf[p][a]++;
            this.misclas[a].push(this.results[idx][x]);
          }
        }

        var s1 = new Array(this.report_data[idx].num_classes).fill(0);
        var s2 = new Array(this.report_data[idx].num_classes).fill(0);

        for(var i=0; i<conf[0].length; i++){
          for(var j=0; j<conf[0].length; j++){
            s2[i] += conf[i][j];
            s1[i] += conf[j][i];
          }
        }

        for (var i=0; i<conf[0].length; i++){
          if(s2[i] != 0){
            pres[i] = conf[i][i]/s2[i];
          }else{
            pres[i] = 0;
          }
          if(s1[i] != 0){
            reca[i] = conf[i][i]/s1[i];
          }else{
            reca[i] = 0;
          }
          if(pres[i]+reca[i] != 0){
            f1[i] = 2*pres[i]*reca[i]/(pres[i]+reca[i]);
          }else{
            f1[i] = 0;
          }
          spec[i] = (corr-conf[i][i])/(s2[i]+corr-2*conf[i][i])
          posl[i] = spec[i]/(1-spec[i])
          negl[i] = (1-spec[i])/spec[i]
          bcrt[i] = 0.5*(reca[i]+spec[i])
          bert[i] = 1-bcrt[i]
          matc[i] = (conf[i][i]*(corr-conf[i][i]) - (s2[i]-conf[i][i])*(s1[i]-conf[i][i]))/Math.sqrt((s2[i])*(s1[i])*(corr+s2[i]-2*conf[i][i])*(corr+s1[i]-2*conf[i][i]))

          this.tabledata.push({'pres': pres[i], 'reca': reca[i], 'f1': f1[i], 'spec': spec[i], 'posl': posl[i], 'negl': negl[i], 'bcrt': bcrt[i], 'bert': bert[i], 'matc': matc[i]});
        }
        console.log(100*corr/(corr+wrng))
        this.acc.push((100*corr/(corr+wrng)).toFixed(2));
        macro.f1 = (average(f1)).toFixed(3);
        macro.prec = (average(pres)).toFixed(2);
        macro.reca = (average(reca)).toFixed(2);
        macro.spec = (average(spec)).toFixed(2);
        macro.posl = (average(posl)).toFixed(2);
        macro.negl = (average(negl)).toFixed(2);
        macro.bcrt = (average(bcrt)).toFixed(2);
        macro.bert = (average(bert)).toFixed(2);
        macro.matc = (average(matc)).toFixed(2);

        this.report_data[idx].macro = macro;
        this.report_data[idx].acc = this.acc;
        this.report_data[idx].misclas = this.misclas;
        this.report_data[idx].tabledata = this.tabledata;
      }
    },
    mounted(){
      this.model_param.num_classes = 43;
      this.num_classes = 43;
      for(var i=0; i<this.$store.state.num_classes; i++){
        this.page[i] = 1;
      }
      var _this = this;
      axios.get(_this.$store.state.server + '/modelinfo').then(function (response){
          for(var x in response.data.result){
            _this.items1.push(response.data.result[x].slice(0,-4))
          }
      })
      axios.get(_this.$store.state.server + '/eval_stats_clear');
      this.$store.commit('updateEvalProg', 0);
    },
    watch: {
      model: function(){
        this.getModelInfo();
      }
    }
  }
</script>
