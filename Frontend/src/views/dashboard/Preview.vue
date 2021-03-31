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
    Browse training, test and validation images for all available datasets
  </v-alert>
  <br/>
  <h4>Select dataset to view:</h4>
  <br/>
  <v-row>
    <v-col cols="12" md="4">
      <v-select
        v-model="sel1"
        :items="datasets"
        label="Select Dataset"
        solo
      ></v-select>
    </v-col>
    <v-col cols="12" md="4">
      <v-select
        v-model="sel"
        :items="items"
        label="Select Dataset"
        solo
      ></v-select>
    </v-col>
  </v-row>
  <v-card>
  <v-tabs
    dark
    background-color="teal darken-3"
    show-arrows
  >
    <v-tabs-slider color="teal lighten-3"></v-tabs-slider>

    <v-tab
      v-for="(i, idx) in nclasses"
      :key="i"
    >
      {{ $store.state.class_labels[idx] }}
    </v-tab>
    <v-tab-item
        v-if="origData && !wait"
        v-for="(n, idx1) in nclasses"
        :key="n"
      >
        <v-container fluid>
          <v-row>
            <v-col
              v-if="origData"
              v-for="(i, idx2) in (origData[idx1].length - (page[idx1]-1)*60 < 60) ? origData[idx1].length - (page[idx1]-1)*60 : 60"
              :key="idx2"
              cols="12"
              sm="1"
            >
              <v-img
                v-if="sel == 'Training Dataset'"
                :src="$store.state.server + '/static/' + ((sel1=='GTSRB Dataset') ? ('Base_48/Train/' + idx1 + '/') : (sel1=='GTSRB_48 Dataset') ? ('Base_48/Train/' + idx1 + '/') : (sel1=='Main Dataset') ? '' : ('Main/Train/' + idx1 + '/')) + origData[idx1][60*(page[idx1]-1) + idx2] + '?ver=' + date"
                aspect-ratio="1"
              ></v-img>
              <v-img
                v-if="sel == 'Test Dataset'"
                :src="$store.state.server + '/static/' + ((sel1=='GTSRB Dataset') ? ('Base_48/Test/' + idx1 + '/') : (sel1=='GTSRB_48 Dataset') ? ('Base_48/Test/' + idx1 + '/') : (sel1=='Main Dataset') ? '' : ('Main/Test/' + idx1 + '/')) + origData[idx1][60*(page[idx1]-1) + idx2] + '?ver=' + date"
                aspect-ratio="1"
              ></v-img>
              <v-img
                v-if="sel == 'Validation Dataset'"
                :src="$store.state.server + '/static/' + ((sel1=='GTSRB Dataset') ? ('Base_48/Val/' + idx1 + '/') : (sel1=='GTSRB_48 Dataset') ? ('Base_48/Val/' + idx1 + '/') : (sel1=='Main Dataset') ? '' : ('Main/Val/' + idx1 + '/')) + origData[idx1][60*(page[idx1]-1) + idx2] + '?ver=' + date"
                aspect-ratio="1"
              ></v-img>
            </v-col>
          </v-row>
          <v-row justify="center">
            <v-col cols="8">
              <v-container class="max-width">
                <v-pagination
                  v-if="origData"
                  v-model="page[idx1]"
                  class="my-4"
                  :length="(origData[idx1].length%60 == 0) ? parseInt(origData[idx1].length/60) : parseInt(origData[idx1].length/60) + 1"
                ></v-pagination>
              </v-container>
            </v-col>
          </v-row>
        </v-container>
      </v-tab-item>
  </v-tabs>
</v-card>
  </v-container>
</template>

<style>

</style>

<script>
  import axios from 'axios';

  export default {
    name: 'Preview',
    components: {

    },
    data () {
      return {
        sel: "Training Dataset",
        sel1: "Main Dataset",
        datasets: ["Main Dataset", "GTSRB Dataset", "GTSRB_48 Dataset", "Difficult Dataset"],
        items: ["Training Dataset", "Test Dataset", "Validation Dataset"],
        page: [],
        origData: null,
        nclasses: 43,
        wait: false
      }
    },
    methods: {
      getOrigInfo(){
        for(var i=0; i<48; i++){
          this.page[i] = 1;
        }
        var _this = this;
        if(this.sel1 == "GTSRB Dataset"){
          this.nclasses = 43;
        }else{
          this.nclasses = 48;
        }
        axios.post(_this.$store.state.server + '/get_dataset', {
          dataset: _this.sel1,
          set: _this.sel
        }).then(response => {
          _this.origData = response.data;
          _this.wait = false;
        })
      }
    },
    computed: {
      date: function(){
        var v = new Date()
        return v.getTime();
      }
    },
    mounted(){
      for(var i=0; i<48; i++){
        this.page[i] = 1;
      }
      this.getOrigInfo();
    },
    watch: {
      sel: function(){
        this.wait = true;
        this.getOrigInfo();
      },
      sel1: function(){
        this.wait = true;
        this.getOrigInfo();
      }
    }
  }
</script>
