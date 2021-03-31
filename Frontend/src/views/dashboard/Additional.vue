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
    View additional images uploaded - apply further transformations, move or delete specific images
  </v-alert>
  <br/>
  <h3>Click on a particular image to add custom transformations or delete it.</h3>
  <br/>
  <v-select
    v-model="sel"
    :items="items"
    label="Select Dataset"
    solo
  ></v-select>
  <v-row>
    <v-col
      cols="12"
      md="4"
    >
    <v-checkbox
      style="margin:0px"
      v-model="multiple"
      hide-details
    >
      <span slot="label" style="color:#000">Select Multiple</span>
    </v-checkbox>
    </v-col>
    <v-col
      cols="12"
      md="4"
    />
    <v-col
      class="text-right"
      cols="12"
      md="4"
      v-if="multiple"
    >
      <v-btn
        color="primary"
        @click="delmulti()"
      >Delete</v-btn>
      <v-menu offset-y>
        <template v-slot:activator="{ on, attrs }">
          <v-btn
            color="primary"
            v-bind="attrs"
            v-on="on"
          >Move to</v-btn>
        </template>
        <v-list>
          <v-list-item v-if="sel!='Training Dataset'" @click="movemulti(0)">
            <v-list-item-title>Training Dataset</v-list-item-title>
          </v-list-item>
          <v-list-item v-if="sel!='Test Dataset'" @click="movemulti(1)">
            <v-list-item-title>Test Dataset</v-list-item-title>
          </v-list-item>
          <v-list-item v-if="sel!='Validation Dataset'" @click="movemulti(2)">
            <v-list-item-title>Validation Dataset</v-list-item-title>
          </v-list-item>
        </v-list>
      </v-menu>
    </v-col>
  </v-row>
  <v-row justify="space-around">
    <v-col cols="auto">
      <v-dialog
        v-model="dialog"
        transition="dialog-bottom-transition"
        max-width="600"
      >
        <template v-slot:default="dialog">
          <v-card>
            <center>
              <br/>
              <v-img
                contain
                :aspect-ratio="1"
                width="500"
                :src="sel_image + '?ver=' + date"
              />
            </center>
            <v-card-actions>
              <router-link to="/edit">
                <v-btn
                  text
                  @click="dialog.value = false; $store.commit('selImage', sel_image); $store.commit('selType', sel)"
                >Edit</v-btn>
              </router-link>
              <div class="text-center">
                <v-menu offset-y>
                  <template v-slot:activator="{ on, attrs }">
                    <v-btn
                      text
                      v-bind="attrs"
                      v-on="on"
                    >Move to</v-btn>
                  </template>
                  <v-list>
                    <v-list-item v-if="sel!='Training Dataset'" @click="move(0)">
                      <v-list-item-title>Training Dataset</v-list-item-title>
                    </v-list-item>
                    <v-list-item v-if="sel!='Test Dataset'" @click="move(1)">
                      <v-list-item-title>Test Dataset</v-list-item-title>
                    </v-list-item>
                    <v-list-item v-if="sel!='Validation Dataset'" @click="move(2)">
                      <v-list-item-title>Validation Dataset</v-list-item-title>
                    </v-list-item>
                  </v-list>
                </v-menu>
              </div>
              <v-btn
                text
                @click="dialog.value = false; delImage()"
              >Delete</v-btn>
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
  <v-card style="margin-top:0">
  <v-tabs
    dark
    background-color="teal darken-3"
    show-arrows
    grow
    v-model="selectedTab"
  >
    <v-tabs-slider color="teal lighten-3"></v-tabs-slider>

    <v-tab
      v-for="(i, idx) in $store.state.num_classes"
      :key="idx"
    >
      {{ $store.state.class_labels[idx] }}
    </v-tab>
    <v-tab-item
        v-if="origData && !wait"
        v-for="(n, idx1) in $store.state.num_classes"
        :key="n"
      >
        <v-container fluid>
          <center v-if="origData[idx1].length==0"><h2><br/>No Image Added</h2></center>
          <v-row>
            <v-col
              v-if="origData"
              v-for="(i, idx2) in (origData[idx1].length - (page[idx1]-1)*36 < 36) ? origData[idx1].length - (page[idx1]-1)*36 : 36"
              :key="i"
              cols="12"
              sm="2"
            >
            <a href="javascript:void(0)" v-if="!multiple">
              <v-img
                @click="sel_idx1 = idx1; sel_idx2 = 36*(page[idx1]-1) + idx2;sel_image = $store.state.server + '/static/Additional/' + ((sel == 'Training Dataset') ? 'Train/' : (sel == 'Test Dataset') ? 'Test/' : 'Val/') + idx1 + '/' + origData[idx1][36*(page[idx1]-1) + idx2];dialog=true"
                :src="$store.state.server + '/static/Additional/' + ((sel == 'Training Dataset') ? 'Train/' : (sel == 'Test Dataset') ? 'Test/' : 'Val/') + idx1 + '/' + origData[idx1][36*(page[idx1]-1) + idx2] + '?ver=' + date"
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
            <a href="javascript:void(0)" v-else>
              <v-img
                @click="changeMulti(idx1, 36*(page[idx1]-1) + idx2)"
                :src="$store.state.server + '/static/Additional/' + ((sel == 'Training Dataset') ? 'Train/' : (sel == 'Test Dataset') ? 'Test/' : 'Val/') + idx1 + '/' + origData[idx1][36*(page[idx1]-1) + idx2] + '?ver=' + date"
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
              <v-fade-transition>
                <v-overlay
                  v-if="multi[idx1][36*(page[idx1]-1) + idx2]"
                  absolute
                  color="#036358"
                >
                  <h4>Selected</h4>
                </v-overlay>
              </v-fade-transition>
              </v-img>
            </a>
            </v-col>
          </v-row>
          <v-row justify="center">
            <v-col cols="8">
              <v-container class="max-width">
                <v-pagination
                  v-if="origData"
                  v-model="page[idx1]"
                  class="my-4"
                  :length="(origData[idx1].length%36 == 0) ? parseInt(origData[idx1].length/36) : parseInt(origData[idx1].length/36) + 1"
                ></v-pagination>
              </v-container>
            </v-col>
          </v-row>
        </v-container>
      </v-tab-item>
  </v-tabs>
</v-card>
<v-row justify="space-around">
  <v-col cols="auto">
    <v-dialog
      v-model="dialog1"
      transition="dialog-bottom-transition"
      max-width="600"
    >
      <template v-slot:default="dialog1">
        <v-card>
          <center>
            <br/>
            <h3>Clusters detected</h3>
            <v-img
              contain
              :aspect-ratio="1.5"
              :src="'data:image/png;base64,' + $store.state.tsne_plot"
            />
          </center>
          <v-card-actions>
            <v-btn
              text
              @click="dialog1.value = false"
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
    name: 'Additional',
    components: {

    },
    data () {
      return {
        items: ["Training Dataset", "Test Dataset", "Validation Dataset"],
        sel_image: "",
        sel_idx1: null,
        sel_idx2: null,
        dialog: false,
        page: [],
        origData: null,
        wait: false,
        multiple: false,
        multi: []
      }
    },
    methods: {
      changeMulti(idx1, idx2){
        var x = this.multi[idx1]
        x[idx2] = !x[idx2]
        this.$set(this.multi, idx1, x);
      },
      delmulti(){
        var _this = this;
        var reqs = []
        for(var x in _this.origData){
          for(var y=0; y<_this.origData[x].length; y++){
            if(_this.multi[x][y]){
              reqs.push(_this.$store.state.server + '/static/Additional/' + ((_this.sel == 'Training Dataset') ? 'Train/' : (_this.sel == 'Test Dataset') ? 'Test/' : 'Val/') + x + '/' + _this.origData[x][y])
            }
          }
        }
        axios.post(_this.$store.state.server + '/delmulti', {
            files: reqs,
            type: _this.sel
        }).then(function (response){
            _this.$notify({title: 'Successful', type: 'success', text: response.data})
            _this.$router.go();
        }).catch(function (error){
            _this.$notify({title: 'Error', type: 'error', text: error.message})
        });
      },
      movemulti(x){
        var _this = this;
        var toType = '';
        if(x == 0){
          toType = "Train";
        }else if(x == 1){
          toType = "Test"
        }else{
          toType = "Val"
        }
        var reqs = []
        for(var x in _this.origData){
          for(var y=0; y<_this.origData[x].length; y++){
            if(_this.multi[x][y]){
              reqs.push(_this.$store.state.server + '/static/Additional/' + ((_this.sel == 'Training Dataset') ? 'Train/' : (_this.sel == 'Test Dataset') ? 'Test/' : 'Val/') + x + '/' + _this.origData[x][y])
            }
          }
        }
        axios.post(_this.$store.state.server + '/movemulti', {
            files: reqs,
            type: _this.sel,
            toType: toType
        }).then(function (response){
            _this.$notify({title: 'Successful', type: 'success', text: response.data})
            _this.$router.go();
        }).catch(function (error){
            _this.$notify({title: 'Error', type: 'error', text: error.message})
        });
      },
      move(x){
        var _this = this;
        var toType = '';
        if(x == 0){
          toType = "Train";
        }else if(x == 1){
          toType = "Test"
        }else{
          toType = "Val"
        }
        axios.post(_this.$store.state.server + '/moveImages', {
            file: _this.sel_image,
            type: _this.sel,
            toType: toType
        }).then(function (response){
            _this.$notify({title: 'Successful', type: 'success', text: response.data})
            _this.origData[_this.sel_idx1].splice(_this.sel_idx2, 1);
            _this.dialog = false;
        }).catch(function (error){
            _this.$notify({title: 'Error', type: 'error', text: error.message})
        });
      },
      delImage(){
        var _this = this;
        axios.post(_this.$store.state.server + '/delImages', {
            file: _this.sel_image,
            type: _this.sel
        }).then(function (response){
            _this.$notify({title: 'Successful', type: 'success', text: response.data})
            _this.origData[_this.sel_idx1].splice(_this.sel_idx2, 1);
            _this.dialog = false;
        }).catch(function (error){
            _this.$notify({title: 'Error', type: 'error', text: error.message})
        });
      },
      getOrigInfo(){
        var _this = this;
        if(this.sel == "Training Dataset"){
          axios.get(_this.$store.state.server + '/addImages/Train').then(response => {
            _this.origData = response.data;
            _this.multi = [];
            for(var x in _this.origData){
              _this.multi[x] = []
              for(var y=0; y<_this.origData[x].length; y++){
                _this.multi[x].push(false)
              }
            }
            _this.wait = false;
          })
        }else if(this.sel == "Test Dataset"){
          axios.get(_this.$store.state.server + '/addImages/Test').then(response => {
            _this.origData = response.data;
            _this.multi = [];
            for(var x in _this.origData){
              _this.multi[x] = []
              for(var y=0; y<_this.origData[x].length; y++){
                _this.multi[x].push(false)
              }
            }
            _this.wait = false;
          })
        }else{
          axios.get(_this.$store.state.server + '/addImages/Val').then(response => {
            _this.origData = response.data;
            _this.multi = [];
            for(var x in _this.origData){
              _this.multi[x] = []
              for(var y=0; y<_this.origData[x].length; y++){
                _this.multi[x].push(false)
              }
            }
            _this.wait = false;
          })
        }
      }
    },
    computed: {
      date: function(){
        var v = new Date()
        return v.getTime();
      },
      sel: {
        get(){
          return this.$store.state.sel;
        },
        set(value){
          this.wait = true;
          this.$store.commit('updateSel', value)
        }
      },
      selectedTab: {
        get(){
          return this.$store.state.selectedTab
        },
        set(value){
          this.$store.commit('updateTab', value)
        }
      },
      dialog1: {
        get(){
          return this.$store.state.show_plot
        },
        set(value){
          this.$store.commit('setplot', value)
        }
      }
    },
    mounted(){
      for(var i=0; i<this.$store.state.num_classes; i++){
        this.page[i] = 1;
      }
      this.getOrigInfo();
    },
    watch: {
      sel: function(){
        this.getOrigInfo();
      }
    }
  }
</script>
