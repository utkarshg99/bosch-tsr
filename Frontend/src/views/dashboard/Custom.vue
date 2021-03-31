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
    Run an existing model on test images
  </v-alert>
  <br/>
  <div v-if="vis">
    <h3>Upload images to predict their class:</h3>
    <br/>
      <FilePond
        name="images"
        ref="pond"
        v-bind:allow-multiple="true"
        accepted-file-types="image/jpeg, image/png"
        v-bind:files="myFiles"
      />
    <br/>
    <center>
      <v-col cols="12" md="4">
        <v-select
          :items="items1"
          v-model="model"
          label="Model to use"
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
      <br/>
      <v-btn
        style="margin-right:0px"
        color="primary"
        @click="evaluateImages()"
      >
        Predict
      </v-btn>
    </center>
  </div>
  <div v-else>
    <br/>
    <v-row>
    <v-col
      v-for="(n,idx) in data"
      :key="idx"
      class="d-flex child-flex"
      cols="12"
      md="3"
    >
      <div class="d-flex flex-column">
      <v-img
        :src="'data:image/png;base64,' + n['Image']"
        aspect-ratio="1"
        class="grey lighten-2"
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
      <center>
        <br/>
        <h3>Predicted Class: <strong>{{$store.state.class_labels[n["Pred"]]}}</strong><br/>Confidence: <strong>{{(n["Confidence"]*100).toFixed(2)}}%</strong></h3>
      </center>
    </div>
    </v-col>
  </v-row>
  <center>
    <v-btn
      color="primary"
      @click="$router.go()"
    >
      Predict More Images
    </v-btn>
  </center>
  </div>
  </v-container>
</template>

<style>
.filepond--panel-root {
    background-color: #a1a1a1;
}
@media (min-width: 30em) {
  .filepond--item {
      width: calc(50% - .5em);
  }
}

@media (min-width: 50em) {
  .filepond--item {
      width: calc(20.00% - .5em);
  }
}
</style>

<script>
import axios from 'axios';

import vueFilePond, { setOptions } from 'vue-filepond';
import 'filepond/dist/filepond.min.css';
// Import image preview and file type validation plugins
import FilePondPluginFileValidateType from "filepond-plugin-file-validate-type";

  export default {
    name: 'DashboardDashboard',
    components: {
      FilePond: vueFilePond(
        FilePondPluginFileValidateType
      )
    },
    data () {
      return {
        data: null,
        vis: true,
        myFiles: [],
        items1: ['Benchmark Model'],
        model: null,
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
      }
    },
    methods: {
      evaluateImages(){
        var _this = this;
        this.$store.commit('load', true);

        if(!this.model){
          _this.$notify({title: 'Error', type: 'error', text: "Please select an option"})
          this.$store.commit('load', false);
        }else{
          var name = _this.model;
          if(name == "Benchmark Model"){
            name = null;
          }
          axios.post(_this.$store.state.server + '/evaluateImages', {
              name: name
          }).then(function (response){
              _this.data = response.data["result"];
              _this.vis = false;
              _this.$store.commit('load', false);
          }).catch(function (error){
              _this.$notify({title: 'Error', type: 'error', text: error.message})
              _this.$store.commit('load', false);
          });
        }
      },
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
    },
    mounted(){
      var _this = this;
      setOptions({
        maxParallelUploads: 15,
        server: {
          process: _this.$store.state.server + "/addImages",
          restore: null,
          load: null,
          fetch: null,
          revert: (uid) => {
            axios.post(_this.$store.state.server + '/addImages', {
                uid: uid
            }, {
              headers: {
                "_method": 'DELETE'
              }
            })
          }
        }
      });

      axios.get(_this.$store.state.server + '/modelinfo').then(function (response){
          for(var x in response.data.result){
            _this.items1.push(response.data.result[x].slice(0,-4))
          }
      })

      axios.get(_this.$store.state.server + '/cleartemp');
    },
    watch: {
      model: function(){
        this.getModelInfo();
      }
    }
  }
</script>
