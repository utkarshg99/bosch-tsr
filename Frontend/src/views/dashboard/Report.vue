<template>
  <v-container
    id="dashboard"
    fluid
    tag="section"
  >
      <center>
        <h2>Evaluation Report</h2>
        <br/>
        <h3>No. of classes model was trained on: {{eval_data.model_param.num_classes}}</h3>
        <h3>No. of classes model was evaluated on: {{eval_data.num_classes}}</h3>
      </center>
      <br/><br/>
      <v-row align="center"
      justify="center">
        <v-col
          cols="12"
          md="6"
        >
          <apexchart type="radialBar" height="400" :options="accChartOptions" :series="eval_data.acc"></apexchart>
        </v-col>
        <v-col
          cols="12"
          md="6"
        >
        <center>
          <h3>Macro Scores</h3>
        </center>
        <br/>
        <v-row>
        <v-col
          cols="12"
          md="4"
        />
        <v-col
          cols="12"
          md="4"
        >
        <center>
          <v-avatar
            color="green"
            size="100"
            style="margin-bottom:10px"
          >
            <span class="white--text headline">{{eval_data.macro.f1}}</span>
          </v-avatar>
          <br/>
          F1 Score
        </center>
        </v-col>
        <v-col
          cols="12"
          md="4"
        />
        <v-col
          cols="12"
          md="3"
        >
        <center>
          <v-avatar
            color="blue"
            size="62"
            style="margin-bottom:10px"
          >
            <span class="white--text headline">{{eval_data.macro.prec}}</span>
          </v-avatar>
          <br/>
          Precision
        </center>
        </v-col>
        <v-col
          cols="12"
          md="3"
        >
        <center>
          <v-avatar
            color="red"
            size="62"
            style="margin-bottom:10px"
          >
            <span class="white--text headline">{{eval_data.macro.reca}}</span>
          </v-avatar>
          <br/>
          Recall
        </center>
        </v-col>
        <v-col
          cols="12"
          md="3"
        >
        <center>
          <v-avatar
            color="purple"
            size="62"
            style="margin-bottom:10px"
          >
            <span class="white--text headline">{{eval_data.macro.spec}}</span>
          </v-avatar>
          <br/>
          Specificity
        </center>
        </v-col>
        <v-col
          cols="12"
          md="3"
        >
        <center>
          <v-avatar
            color="orange"
            size="62"
            style="margin-bottom:10px"
          >
            <span class="white--text headline">{{eval_data.macro.posl}}</span>
          </v-avatar>
          <br/>
          Positive Likelihood
        </center>
        </v-col>
        <v-col
          cols="12"
          md="3"
        >
        <center>
          <v-avatar
            color="orange"
            size="62"
            style="margin-bottom:10px"
          >
            <span class="white--text headline">{{eval_data.macro.negl}}</span>
          </v-avatar>
          <br/>
          Negative Likelihood
        </center>
        </v-col>
        <v-col
          cols="12"
          md="3"
        >
        <center>
          <v-avatar
            color="purple"
            size="62"
            style="margin-bottom:10px"
          >
            <span class="white--text headline">{{eval_data.macro.bcrt}}</span>
          </v-avatar>
          <br/>
          Balanced Classification Rate
        </center>
        </v-col>
        <v-col
          cols="12"
          md="3"
        >
        <center>
          <v-avatar
            color="red"
            size="62"
            style="margin-bottom:10px"
          >
            <span class="white--text headline">{{eval_data.macro.bert}}</span>
          </v-avatar>
          <br/>
          Balanced Error Rate
        </center>
        </v-col>
        <v-col
          cols="12"
          md="3"
        >
        <center>
          <v-avatar
            color="blue"
            size="62"
            style="margin-bottom:10px"
          >
            <span class="white--text headline">{{eval_data.macro.matc}}</span>
          </v-avatar>
          <br/>
          Matthew's Correlation
        </center>
        </v-col>
      </v-row>
        </v-col>
        <v-col
          cols="12"
          md="3"
        >
        <br/>
        <v-select
          v-model="bar_selected"
          :items="bar_select"
          label="Metric to display on chart"
          outlined
        ></v-select>
        </v-col>
        <v-col
          cols="12"
          md="12"
        >
          <apexchart type="bar" ref="bar" height="350" :options="baroptions" :series="bar_data"></apexchart>
        </v-col>
        <v-col
          cols="12"
          md="12"
        >
        <br/>
          <v-simple-table
            fixed-header
            height="350px"
          >
            <template v-slot:default>
              <thead>
                <tr>
                  <th class="text-left">
                    S.No.
                  </th>
                  <th class="text-left">
                    Class
                  </th>
                  <th class="text-left">
                    Precision
                  </th>
                  <th class="text-left">
                    Recall
                  </th>
                  <th class="text-left">
                    Specificity
                  </th>
                  <th class="text-left">
                    Positive Likelihood
                  </th>
                  <th class="text-left">
                    Negative Likelihood
                  </th>
                  <th class="text-left">
                    Balanced Classification Rate
                  </th>
                  <th class="text-left">
                    Balanced Error Rate
                  </th>
                  <th class="text-left">
                    Matthew's Correlation
                  </th>
                  <th class="text-left">
                    F1-Score
                  </th>
                </tr>
              </thead>
              <tbody>
                <tr
                  v-for="(item, idx) in eval_data.tabledata"
                  :key="idx"
                >
                  <td :style="(item.f1 == 0)?'background-color: red; color: #fff':''">{{ idx }}</td>
                  <td :style="(item.f1 == 0)?'background-color: red; color: #fff':''">{{ $store.state.class_labels[idx] }}</td>
                  <td :style="(item.f1 == 0)?'background-color: red; color: #fff':''">{{ item.pres.toFixed(3) }}</td>
                  <td :style="(item.f1 == 0)?'background-color: red; color: #fff':''">{{ item.reca.toFixed(3) }}</td>
                  <td :style="(item.f1 == 0)?'background-color: red; color: #fff':''">{{ item.spec.toFixed(3) }}</td>
                  <td :style="(item.f1 == 0)?'background-color: red; color: #fff':''">{{ item.posl.toFixed(3) }}</td>
                  <td :style="(item.f1 == 0)?'background-color: red; color: #fff':''">{{ item.negl.toFixed(3) }}</td>
                  <td :style="(item.f1 == 0)?'background-color: red; color: #fff':''">{{ item.bcrt.toFixed(3) }}</td>
                  <td :style="(item.f1 == 0)?'background-color: red; color: #fff':''">{{ item.bert.toFixed(3) }}</td>
                  <td :style="(item.f1 == 0)?'background-color: red; color: #fff':''">{{ item.matc.toFixed(3) }}</td>
                  <td :style="(item.f1 == 0)?'background-color: red; color: #fff':''">{{ item.f1.toFixed(3) }}</td>
                </tr>
              </tbody>
            </template>
          </v-simple-table>
        </v-col>
        <v-col
          cols="12"
          md="12"
        >
        <br/>
        <h3>Optimizer & Scheduler options used for training:</h3><br/>
        <v-row>
          <v-col
            cols="12"
            md="3"
          >
          <center>
            <v-avatar
              color="blue"
              size="62"
              style="margin-bottom:10px"
            >
              <span class="white--text">{{eval_data.model_param.lr}}</span>
            </v-avatar>
            <br/>
            Learning Rate
          </center>
          </v-col>
          <v-col
            cols="12"
            md="3"
            v-if="eval_data.model_param.method == 'Train further'"
          >
          <center>
            <v-avatar
              color="red"
              size="62"
              style="margin-bottom:10px"
            >
              <span class="white--text">{{eval_data.model_param.momentum}}</span>
            </v-avatar>
            <br/>
            Momentum
          </center>
          </v-col>
          <v-col
            cols="12"
            md="3"
          >
          <center>
            <v-avatar
              color="purple"
              size="62"
              style="margin-bottom:10px"
            >
              <span class="white--text">{{eval_data.model_param.gamma}}</span>
            </v-avatar>
            <br/>
            Gamma
          </center>
          </v-col>
          <v-col
            cols="12"
            md="3"
            v-if="eval_data.model_param.method == 'Train further'"
          >
          <center>
            <v-avatar
              color="orange"
              size="62"
              style="margin-bottom:10px"
            >
              <span class="white--text">{{eval_data.model_param.l2_norm}}</span>
            </v-avatar>
            <br/>
            L2 Normalization
          </center>
          </v-col>
        </v-row>
        </v-col>
        <v-col
          cols="12"
          md="12"
        >
        <center><br/><h3>Misclassified Images</h3><p>Click on image to know details</p></center>
        <v-card>
          <v-tabs
            dark
            background-color="teal darken-3"
            show-arrows
          >
            <v-tabs-slider color="teal lighten-3"></v-tabs-slider>

            <v-tab
              v-for="(id1, i) in eval_data.num_classes"
              :key="i"
            >
              {{ $store.state.class_labels[i] }}
            </v-tab>
            <v-tab-item
                v-if="misclas"
                v-for="(x, n) in misclas"
                :key="n"
              >
              <v-container fluid>
                <center v-if="x.length==0"><h2><br/>No Image Misclassified</h2></center>
                <v-row v-else>
                <v-col
                  v-for="(i, idx) in (x.length - (page[n]-1)*36 < 36) ? x.length - (page[n]-1)*36 : 36"
                  :key="idx"
                  cols="12"
                  md="2"
                >
                  <a href="javascript:void(0)">
                    <v-img
                      :src="$store.state.server + '/' + x[36*(page[n]-1) + idx]['Location'] + '?ver=' + date"
                      @click="sel_image = idx; sel_class = n; heatmap()"
                      aspect-ratio="1"
                    >
                    <template v-slot:placeholder>
                      <v-row
                        class="fill-height ma-0"
                        align="center"
                        justify="center"
                      >
                        <v-progress-circular
                          indeterminate
                          color="grey lighten-5"
                        ></v-progress-circular>
                      </v-row>
                    </template>
                    </v-img>
                  </a>
                  </v-col>
                </v-row>
                <v-row justify="center">
                  <v-col cols="8">
                    <v-container class="max-width">
                      <v-pagination
                        v-if="misclas"
                        v-model="page[n]"
                        class="my-4"
                        :length="(x.length%36 == 0) ? parseInt(x.length/36) : parseInt(x.length/36) + 1"
                      ></v-pagination>
                    </v-container>
                  </v-col>
                </v-row>
              </v-container>
            </v-tab-item>
          </v-tabs>
        </v-card>
        </v-col>
      </v-row>
      <v-row justify="space-around">
        <v-col cols="auto">
          <v-dialog
            v-model="dialog"
            transition="dialog-bottom-transition"
            max-width="1000"
          >
            <template v-slot:default="dialog">
              <v-card>
                <br/>
                <v-row>
                  <v-col cols="12" md="4">
                    <br/>
                    <v-img
                      contain
                      :aspect-ratio="1"
                      :src="'data:image/png;base64,' + heat"
                    />
                  </v-col>
                  <v-col cols="12" md="4">
                    <br/>
                    <v-img
                      contain
                      :aspect-ratio="1"
                      :src="'data:image/png;base64,' + heat1"
                    />
                  </v-col>
                  <v-col cols="12" md="4">
                    <br/>
                    <v-img
                      style="margin:30px"
                      v-if="show_anchor==true"
                      contain
                      :aspect-ratio="1"
                      :src="'data:image/png;base64,' + heat2"
                    />
                    <v-container v-else fill-height>
                      <v-row class="align-center">
                        <v-col cols="12" sm="2"/>
                        <v-col cols="12" sm="4">
                          <v-btn
                            color="primary"
                            @click="anchor()"
                          >
                            Show Anchor
                          </v-btn>
                        </v-col>
                      </v-row>
                    </v-container>
                  </v-col>
                <v-col cols="12" md="12">
                <v-simple-table style="padding-left:20px;padding-right:20px">
                  <template v-slot:default>
                    <tbody>
                      <tr>
                        <td><strong>Location:</strong></td>
                        <td class="text-right">{{misclas[sel_class][sel_image]['Location']}}</td>
                      </tr>
                      <tr>
                        <td><strong>Actual Class:</strong></td>
                        <td class="text-right">{{$store.state.class_labels[misclas[sel_class][sel_image]['Actual']]}}</td>
                      </tr>
                      <tr>
                        <td><strong>Predicted Class:</strong></td>
                        <td class="text-right">{{$store.state.class_labels[misclas[sel_class][sel_image]['Pred']]}}</td>
                      </tr>
                      <tr>
                        <td><strong>Confidence of Prediction:</strong></td>
                        <td class="text-right">{{(misclas[sel_class][sel_image]['Conf']*100).toFixed(2)}} %</td>
                      </tr>
                      <tr>
                        <td><strong>Augmentations Applied:</strong></td>
                        <td class="text-right" v-if="misclas[sel_class][sel_image]['Transformations'].length">{{misclas[sel_class][sel_image]['Transformations'].join()}}</td>
                        <td class="text-right" v-else>None</td>
                      </tr>
                    </tbody>
                  </template>
                </v-simple-table>
                </v-col>
                </v-row>
                <v-card-actions>
                  <v-btn
                    text
                    @click="dialog.value = false"
                  >Close</v-btn>
                </v-card-actions>
              </v-card>
            </template>
          </v-dialog>
        </v-col>
      </v-row>
  </v-container>
</template>

<style>

</style>

<script>
import axios from 'axios';

  export default {
    name: 'Report',
    props: {
      eval_data: {
        type: Object,
        default() {
          return {
            model_param: {
              "method": "Train further",
              "num_classes": 43,
              "epochs": 5,
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
            model: "Benchmark Model",
            dataset: "Main dataset",
            acc: [0],
            misclas: [],
            tabledata: [],
            num_classes: 43
          }
        }
      }
    },
    data () {
      return {
        misclas: null,
        sel_image: null,
        sel_class: null,
        page: [],
        dialog: false,
        heat: null,
        heat1: null,
        heat2: null,
        show_anchor: false,
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
        bar_selected: null,
        bar_select: [
          {
            text: 'F1 Score',
            value: 'f1'
          },
          {
            text: 'Precision',
            value: 'pres'
          },
          {
            text: 'Recall',
            value: 'reca'
          },
          {
            text: 'Specificity',
            value: 'spec'
          },
          {
            text: 'Negative Likelihood',
            value: 'negl'
          },
          {
            text: 'Balanced Classification Rate',
            value: 'bcrt'
          },
          {
            text: 'Balanced Error Rate',
            value: 'bert'
          },
          {
            text: "Matthew's Correlation",
            value: 'matc'
          },
        ],
        bar_data: [{
            name: 'Value',
            data: []
        }],
        baroptions: {
            chart: {
              type: 'bar',
              height: 350
            },
            plotOptions: {
              bar: {
                horizontal: false,
                columnWidth: '55%',
                endingShape: 'rounded'
              },
            },
            dataLabels: {
              enabled: false
            },
            stroke: {
              show: true,
              width: 2,
              colors: ['transparent']
            },
            xaxis: {
              categories: [],
              title: {
                text: 'Class'
              },
            },
            yaxis: {
              title: {
                text: 'Value'
              }
            },
            fill: {
              opacity: 1
            }
          },
      }
    },
    methods: {
      heatmap(){
        var _this = this;
        this.$store.commit('load', true);
        var name = _this.eval_data.model;
        if(name == "Benchmark Model"){
          name = null;
        }
        axios.post(_this.$store.state.server + '/heatmap', {
            loc: _this.misclas[_this.sel_class][_this.sel_image]['Location'],
            name: name,
            cap: true
        }).then(function (response){
            _this.heat = response.data.blended;
            _this.heat1 = response.data.normal;
            _this.show_anchor = false;
            _this.dialog = true;
            _this.$store.commit('load', false);
        }).catch(function (error){
            _this.$notify({title: 'Error', type: 'error', text: error.message})
            _this.$store.commit('load', false);
        });
      },
      anchor(){
        var _this = this;
        this.$store.commit('load', true);
        var name = _this.eval_data.model;
        if(name == "Benchmark Model"){
          name = null;
        }
        axios.post(_this.$store.state.server + '/heatmap', {
            loc: _this.misclas[_this.sel_class][_this.sel_image]['Location'],
            name: name,
            cap: false
        }).then(function (response){
            _this.heat2 = response.data;
            _this.show_anchor = true;
            _this.$store.commit('load', false);
        }).catch(function (error){
            _this.$notify({title: 'Error', type: 'error', text: error.message})
            _this.$store.commit('load', false);
        });
      }
    },
    computed: {
      date: function(){
        var v = new Date()
        return v.getTime();
      },
    },
    mounted(){
      this.bar_selected = 'f1';
      var xaxis = [];
      for(var i=0; i<this.$store.state.num_classes; i++){
        this.page[i] = 1;
      }
      for(var i=0; i<this.eval_data.num_classes; i++){
        xaxis.push(i.toString())
      }
      this.misclas = this.eval_data.misclas;
      this.$refs.bar.updateOptions({
        xaxis: {
          categories: xaxis
        }
      })
    },
    watch: {
      bar_selected: function(val){
        var data = [];
        for(var x=0; x<this.eval_data.tabledata.length; x++){
          data.push(this.eval_data.tabledata[x][val].toFixed(3));
        }
        this.$refs.bar.updateSeries([
          {
            name: 'Value',
            data: data
          }
        ])
      }
    }
  }
</script>
