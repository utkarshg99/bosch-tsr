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
    Upload images for registered labels with options for augmentation and segregation
  </v-alert>
  <br/>
  <v-stepper
    v-model="e6"
    vertical
  >
    <v-stepper-step
      :complete="e6 > 1"
      step="1"
    >
      Select class of images to be added
    </v-stepper-step>

    <v-stepper-content step="1">
      <p><strong>NOTE:</strong> New classes introduced will appear at the bottom of the list</p>
      <v-select
          v-model="sel_class"
          :items="items"
          label="Select a class"
        ></v-select>
      <v-btn
        v-if="sel_class!=null"
        color="primary"
        @click="e6 = 2"
      >
        Next
      </v-btn>
    </v-stepper-content>

    <v-stepper-step
      :complete="e6 > 2"
      step="2"
    >
      Configure Dataset Options
    </v-stepper-step>

    <v-stepper-content step="2">
      <h4>Select Augmentations & Transformations to apply (if any): </h4>
      <v-row>
        <v-col
          cols="12"
          md="3"
        >
        <v-checkbox
            v-model="tabs['Brightness & Contrast']"
            color="primary"
            hide-details
          >
          <span slot="label" style="color:#000">Brightness & Contrast</span>
        </v-checkbox>
        </v-col>

        <v-col
          cols="12"
          md="3"
        >
        <v-checkbox
            v-model="tabs['Rotate']"
            color="primary"
            hide-details
          >
          <span slot="label" style="color:#000">Shift & Rotate</span>
        </v-checkbox>
        </v-col>

        <v-col
          cols="12"
          md="3"
        >
        <v-checkbox
            v-model="tabs['Blur & Distort']"
            color="primary"
            hide-details
          >
          <span slot="label" style="color:#000">Blur & Distort</span>
        </v-checkbox>
        </v-col>

        <v-col
          cols="12"
          md="3"
        >
        <v-checkbox
            v-model="tabs['Noise']"
            color="primary"
            hide-details
          >
          <span slot="label" style="color:#000">Noise</span>
        </v-checkbox>
        </v-col>

        <v-col
          cols="12"
          md="3"
        >
        <v-checkbox
            v-model="tabs['Hue & Saturation']"
            color="primary"
            hide-details
          >
          <span slot="label" style="color:#000">Hue & Saturation</span>
        </v-checkbox>
        </v-col>

        <v-col
          cols="12"
          md="3"
        >
        <v-checkbox
            v-model="tabs['Dropout & Cutout']"
            color="primary"
            hide-details
          >
          <span slot="label" style="color:#000">Dropout & Cutout</span>
        </v-checkbox>
        </v-col>

        <v-col
          cols="12"
          md="3"
        >
        <v-checkbox
            v-model="tabs['Affine & Perspective']"
            color="primary"
            hide-details
          >
          <span slot="label" style="color:#000">Affine & Perspective transform</span>
        </v-checkbox>
        </v-col>
      </v-row>
      <div v-if="tabs['Noise']||tabs['Brightness & Contrast']||tabs['Rotate']||tabs['Blur & Distort']||tabs['Hue & Saturation']||tabs['Dropout & Cutout']||tabs['Affine & Perspective']">
        <h4><br/><br/>Number of augmented images to generate per input image:</h4>
        <br/>
        <v-slider
          style="margin-top:10px"
          v-model="num_images"
          :thumb-size="24"
          thumb-label="always"
          step="10"
          ticks
        >
          <template v-slot:thumb-label="{ value }">
            {{ blur_choices[Math.min(Math.floor(value / 10), 9)] }}
          </template>
        </v-slider>
        <h4>Select options for selected augmentations & transformations:</h4>
      </div>
      <v-card>
        <v-tabs
          v-if="tabs['Noise']||tabs['Brightness & Contrast']||tabs['Rotate']||tabs['Blur & Distort']||tabs['Hue & Saturation']||tabs['Dropout & Cutout']||tabs['Affine & Perspective']"
          dark
          centered
          background-color="teal darken-3"
          show-arrows
        >
          <v-tabs-slider color="teal lighten-3"></v-tabs-slider>
          <v-tab v-if="tabs['Brightness & Contrast']">
            Brightness & Contrast
          </v-tab>
          <v-tab v-if="tabs['Rotate']">
            Shift & Rotate
          </v-tab>
          <v-tab v-if="tabs['Blur & Distort']">
            Blur & Distort
          </v-tab>
          <v-tab v-if="tabs['Noise']">
            Noise
          </v-tab>
          <v-tab v-if="tabs['Hue & Saturation']">
            Hue & Saturation
          </v-tab>
          <v-tab v-if="tabs['Dropout & Cutout']">
            Dropout & Cutout
          </v-tab>
          <v-tab v-if="tabs['Affine & Perspective']">
            Affine & Perspective Transform
          </v-tab>
          <v-tab-item v-if="tabs['Brightness & Contrast']" style="margin-left:20px; margin-right:20px">
            <center>
              <br/>
              <v-row>
                <v-col
                  cols="12"
                  md="6"
                >
                <v-subheader class="pl-0">
                  Probability of Brightness Change (in %)
                </v-subheader>
                <v-slider
                  v-model="brightness"
                  thumb-label
                  class="align-center"
                ></v-slider>
                </v-col>

                <v-col
                  cols="12"
                  md="6"
                >
                <v-subheader class="pl-0">
                  Probability of Contrast Change (in %)
                </v-subheader>
                <v-slider
                  v-model="contrast"
                  thumb-label
                  class="align-center"
                ></v-slider>
                </v-col>
                <v-col
                  cols="12"
                  md="6"
                >
                  <v-subheader class="pl-0">
                    Limits for Brightness Change
                  </v-subheader>
                  <v-range-slider
                    v-model="range_brightness"
                    max="100"
                    min="-100"
                    hide-details
                    class="align-center"
                  >
                    <template v-slot:prepend>
                      <v-text-field
                        :value="range_brightness[0]/100.0"
                        class="mt-0 pt-0"
                        hide-details
                        single-line
                        type="number"
                        style="width: 60px"
                        @change="$set(range_brightness, 0, $event)"
                      ></v-text-field>
                    </template>
                    <template v-slot:append>
                      <v-text-field
                        :value="range_brightness[1]/100.0"
                        class="mt-0 pt-0"
                        hide-details
                        single-line
                        type="number"
                        style="width: 60px"
                        @change="$set(range_brightness, 1, $event)"
                      ></v-text-field>
                    </template>
                  </v-range-slider>
                </v-col>
                <v-col
                  cols="12"
                  md="6"
                >
                  <v-subheader class="pl-0">
                    Limits for Contrast Change
                  </v-subheader>
                  <v-range-slider
                    v-model="range_contrast"
                    max="100"
                    min="-100"
                    hide-details
                    class="align-center"
                  >
                    <template v-slot:prepend>
                      <v-text-field
                        :value="range_contrast[0]/100.0"
                        class="mt-0 pt-0"
                        hide-details
                        single-line
                        type="number"
                        style="width: 60px"
                        @change="$set(range_contrast, 0, $event)"
                      ></v-text-field>
                    </template>
                    <template v-slot:append>
                      <v-text-field
                        :value="range_contrast[1]/100.0"
                        class="mt-0 pt-0"
                        hide-details
                        single-line
                        type="number"
                        style="width: 60px"
                        @change="$set(range_contrast, 1, $event)"
                      ></v-text-field>
                    </template>
                  </v-range-slider>
                </v-col>
              </v-row>
            </center>
            <br/>
          </v-tab-item>
          <v-tab-item v-if="tabs['Rotate']" style="margin-left:20px; margin-right:20px">
            <center>
              <v-row>
                <v-col
                  cols="12"
                  md="6"
                >
                <v-subheader class="pl-0">
                  Probability of Shift & Rotate (in %)
                </v-subheader>
                <v-slider
                  v-model="rotate"
                  thumb-label
                  class="align-center"
                ></v-slider>
                </v-col>
                <v-col
                  cols="12"
                  md="6"
                >
                  <v-subheader>Limits on Rotation (in degrees)</v-subheader>
                  <v-card-text>
                    <v-range-slider
                      v-model="range_rotate"
                      max="90"
                      min="-90"
                      hide-details
                      class="align-center"
                    >
                    <template v-slot:prepend>
                      <v-text-field
                        :value="range_rotate[0]"
                        class="mt-0 pt-0"
                        hide-details
                        single-line
                        type="number"
                        style="width: 60px"
                        @change="$set(range_rotate, 0, $event)"
                      ></v-text-field>
                    </template>
                    <template v-slot:append>
                      <v-text-field
                        :value="range_rotate[1]"
                        class="mt-0 pt-0"
                        hide-details
                        single-line
                        type="number"
                        style="width: 60px"
                        @change="$set(range_rotate, 1, $event)"
                      ></v-text-field>
                    </template>
                  </v-range-slider>
                </v-card-text>
              </v-col>
            </v-row>
            </center>
          </v-tab-item>
          <v-tab-item v-if="tabs['Blur & Distort']" style="margin-left:20px; margin-right:20px">
            <center>
              <br/>
              <v-row>
                <v-col
                  cols="12"
                  md="6"
                >
                <v-subheader class="pl-0">
                  Probability of Motion Blur (in %)
                </v-subheader>
                <v-slider
                  v-model="mblur"
                  thumb-label
                  class="align-center"
                ></v-slider>
                </v-col>
                <v-col
                  cols="12"
                  md="6"
                >
                <v-subheader class="pl-0">
                  Probability of Gaussian Blur (in %)
                </v-subheader>
                <v-slider
                  v-model="gblur"
                  thumb-label
                  class="align-center"
                ></v-slider>
                </v-col>
                <v-col
                  cols="12"
                  md="6"
                >
                <v-subheader class="pl-0">
                  Probability of Median Blur (in %)
                </v-subheader>
                <v-slider
                  v-model="mdblur"
                  thumb-label
                  class="align-center"
                ></v-slider>
                </v-col>
                <v-col
                  cols="12"
                  md="6"
                >
                <v-subheader class="pl-0">
                  Probability of Distort (in %)
                </v-subheader>
                <v-slider
                  v-model="distort"
                  thumb-label
                  class="align-center"
                ></v-slider>
                </v-col>
              </v-row>
            </center>
            <br/>
          </v-tab-item>
          <v-tab-item v-if="tabs['Noise']" style="margin-left:20px; margin-right:20px">
            <center>
              <br/>
              <v-row>
                <v-col
                  cols="12"
                  md="6"
                >
                <v-subheader class="pl-0">
                  Probability of Gaussian Noise (in %)
                </v-subheader>
                <v-slider
                  v-model="gnoise"
                  thumb-label
                  class="align-center"
                ></v-slider>
                </v-col>
                <v-col
                  cols="12"
                  md="6"
                >
                <v-subheader class="pl-0">
                  Probability of ISO Noise (in %)
                </v-subheader>
                <v-slider
                  v-model="inoise"
                  thumb-label
                  class="align-center"
                ></v-slider>
                </v-col>
                <v-col
                  cols="12"
                  md="6"
                >
                <v-subheader class="pl-0">
                  Probability of Multiplicative Noise (in %)
                </v-subheader>
                <v-slider
                  v-model="mnoise"
                  thumb-label
                  class="align-center"
                ></v-slider>
                </v-col>
              </v-row>
            </center>
            <br/>
          </v-tab-item>
          <v-tab-item v-if="tabs['Hue & Saturation']" style="margin-left:20px; margin-right:20px">
            <center>
              <br/>
              <v-row>
                <v-col
                  cols="12"
                  md="6"
                >
                <v-subheader class="pl-0">
                  Probability of Change in Hue & Saturation (in %)
                </v-subheader>
                <v-slider
                  v-model="hue"
                  thumb-label
                  class="align-center"
                ></v-slider>
                </v-col>
                <v-col
                  cols="12"
                  md="6"
                >
                <v-subheader class="pl-0">
                  Limits for Hue Shift
                </v-subheader>
                <v-range-slider
                  v-model="range_hue"
                  max="100"
                  min="-100"
                  hide-details
                  class="align-center"
                >
                <template v-slot:prepend>
                  <v-text-field
                    :value="range_hue[0]"
                    class="mt-0 pt-0"
                    hide-details
                    single-line
                    type="number"
                    style="width: 60px"
                    @change="$set(range_hue, 0, $event)"
                  ></v-text-field>
                </template>
                <template v-slot:append>
                  <v-text-field
                    :value="range_hue[1]"
                    class="mt-0 pt-0"
                    hide-details
                    single-line
                    type="number"
                    style="width: 60px"
                    @change="$set(range_hue, 1, $event)"
                  ></v-text-field>
                </template>
              </v-range-slider>
                </v-col>
                <v-col
                  cols="12"
                  md="6"
                >
                <v-subheader class="pl-0">
                  Limits for Saturation Shift
                </v-subheader>
                <v-range-slider
                  v-model="range_sat"
                  max="100"
                  min="-100"
                  hide-details
                  class="align-center"
                >
                <template v-slot:prepend>
                  <v-text-field
                    :value="range_sat[0]"
                    class="mt-0 pt-0"
                    hide-details
                    single-line
                    type="number"
                    style="width: 60px"
                    @change="$set(range_sat, 0, $event)"
                  ></v-text-field>
                </template>
                <template v-slot:append>
                  <v-text-field
                    :value="range_sat[1]"
                    class="mt-0 pt-0"
                    hide-details
                    single-line
                    type="number"
                    style="width: 60px"
                    @change="$set(range_sat, 1, $event)"
                  ></v-text-field>
                </template>
              </v-range-slider>
                </v-col>
                <v-col
                  cols="12"
                  md="6"
                >
                <v-subheader class="pl-0">
                  Probability of Color Jitter (in %)
                </v-subheader>
                <v-slider
                  v-model="jitter"
                  thumb-label
                  class="align-center"
                ></v-slider>
                </v-col>
              </v-row>
            </center>
            <br/>
          </v-tab-item>
          <v-tab-item v-if="tabs['Dropout & Cutout']" style="margin-left:20px; margin-right:20px">
            <center>
              <br/>
              <v-row>
                <v-col
                  cols="12"
                  md="6"
                >
                <v-subheader class="pl-0">
                  Probability of Coarse Dropout (in %)
                </v-subheader>
                <v-slider
                  v-model="dropout"
                  thumb-label
                  class="align-center"
                ></v-slider>
                </v-col>
                <v-col
                  cols="12"
                  md="6"
                >
                <v-subheader class="pl-0">
                  Probability of Cutout (in %)
                </v-subheader>
                <v-slider
                  v-model="cutout"
                  thumb-label
                  class="align-center"
                ></v-slider>
                </v-col>
              </v-row>
            </center>
            <br/>
          </v-tab-item>
          <v-tab-item v-if="tabs['Affine & Perspective']" style="margin-left:20px; margin-right:20px">
            <center>
              <br/>
              <v-row>
                <v-col
                  cols="12"
                  md="6"
                >
                <v-subheader class="pl-0">
                  Probability of Random Affine transform (in %)
                </v-subheader>
                <v-slider
                  v-model="affine"
                  thumb-label
                  class="align-center"
                ></v-slider>
                </v-col>
                <v-col
                  cols="12"
                  md="6"
                >
                <v-subheader class="pl-0">
                  Probability of Random Perspective transform (in %)
                </v-subheader>
                <v-slider
                  v-model="perspective"
                  thumb-label
                  class="align-center"
                ></v-slider>
                </v-col>
              </v-row>
            </center>
            <br/>
          </v-tab-item>
        </v-tabs>
      </v-card>
      <br/>
      <v-checkbox
          v-model="add_test"
          color="primary"
          hide-details
        >
        <span slot="label" style="color:#000">Add images to test dataset also</span>
      </v-checkbox>
      <br/><br/>
      <div v-if="add_test">
        <h4>Select Train:Test:Validation Ratio: </h4>
        <br/>
        <v-range-slider
          v-model="ratio"
          step="5"
          :hint="hint"
          persistent-hint
        ></v-range-slider>
      </div>
      <div v-else>
        <h4>Method of segregation:</h4>
        <v-col cols="12" md="4">
        <v-select
            v-model="seg"
            :items="seg_methods"
          ></v-select>
        </v-col>
        <br/>
        <h4 v-if="seg == 'Smart Segregation'">
          Select Train:Validation ratio to pick from each cluster:
        </h4>
        <h4 v-else>
          Select Train:Validation ratio:
        </h4>
        <br/>
        <v-slider
          v-model="ratio1"
          :thumb-size="24"
          :hint="hint1"
          persistent-hint
        ></v-slider>
      </div>
      <br/>
      <v-btn
        color="primary"
        @click="e6 = 3"
      >
        Next
      </v-btn>
      <v-btn text @click="e6 = 1">
        Back
      </v-btn>
    </v-stepper-content>

    <v-stepper-step
      :complete="e6 > 3"
      step="3"
    >
      Select Images
    </v-stepper-step>

    <v-stepper-content step="3">
      <br/>
        <FilePond
          name="images"
          ref="pond"
          v-bind:allow-multiple="true"
          accepted-file-types="image/jpeg, image/png"
          v-bind:files="myFiles"
        />
      <br/>
      <v-btn
        color="primary"
        @click="e6 = 3; finalAddImages()"
      >
        Add Images
      </v-btn>
      <v-btn text @click="e6 = 2">
        Back
      </v-btn>
    </v-stepper-content>

  </v-stepper>
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
        add_test: false,
        seg: "Smart Segregation",
        seg_methods: ["Smart Segregation", "Random Segregation"],
        num_images: 20,
        blur_choices: ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
        brightness: 10,
        range_brightness: [-30, 30],
        contrast: 10,
        range_contrast: [-30, 30],
        rotate: 10,
        range_rotate: [-30, 30],
        distort: 10,
        mnoise: 10,
        gnoise: 10,
        inoise: 10,
        gblur: 10,
        mblur: 10,
        mdblur: 10,
        jitter: 10,
        dropout: 10,
        cutout: 10,
        affine: 10,
        perspective: 10,
        hue: 10,
        range_hue: [-10, 10],
        range_sat: [-10, 10],
        tabs: {"Noise" : true, "Brightness & Contrast" : true, "Rotate" : true, "Blur & Distort" : true, "Hue & Saturation" : false, "Dropout & Cutout" : false, "Affine & Perspective": false},
        ratio: [50, 80],
        ratio1: 70,
        sel_class: null,
        value: null,
        items: [],
        e6: 1,
        myFiles: [],
      }
    },
    methods: {
      finalAddImages(){
        var _this = this;
        var num = this.num_images/10 + 1;
        if(!this.tabs['Noise']&&!this.tabs['Brightness & Contrast']&&!this.tabs['Rotate']&&!this.tabs['Blur & Distort']&&!this.tabs['Hue & Saturation']&&!this.tabs['Dropout & Cutout']&&!this.tabs['Affine & Perspective']){
          num = 0;
        }
        var ratio = _this.ratio;
        if(!this.add_test){
          ratio = _this.ratio1/100;
        }
        _this.$store.commit('load', true);
        axios.post(_this.$store.state.server + '/finalAddImages', {
            class: _this.sel_class,
            num: num,
            ratio: ratio,
            add_test: _this.add_test,
            seg: _this.seg,
            options: {
              brightness: _this.brightness/100,
              range_brightness: _this.range_brightness.map(x => x/100),
              contrast: _this.contrast/100,
              range_contrast: _this.range_contrast.map(x => x/100),
              rotate: _this.rotate/100,
              range_rotate: _this.range_rotate,
              mblur: _this.mblur/100,
              mdblur: _this.mdblur/100,
              gblur: _this.gblur/100,
              distort: _this.distort/100,
              inoise: _this.inoise/100,
              gnoise: _this.gnoise/100,
              mnoise: _this.mnoise/100,
              hue: _this.hue/100,
              range_hue: _this.range_hue,
              range_sat: _this.range_sat,
              jitter: _this.jitter,
              dropout: _this.dropout,
              cutout: _this.cutout,
              affine: _this.affine,
              perspective: _this.perspective,
              tabs: _this.tabs
            }
        }).then(function (response){
            if(response.data){
              _this.$notify({title: 'Successful', type: 'success', text: "Successfully added images"})
              if(_this.seg == "Smart Segregation"){
                _this.$store.commit('settsne', response.data)
                _this.$store.commit('setplot', true)
              }
              var tabtoshow = parseInt(_this.sel_class.slice(6))
              _this.$store.commit('updateTab', tabtoshow)
              _this.$router.push("images")
            }
            _this.$store.commit('load', false);
        }).catch(function (error){
            _this.$notify({title: 'Error', type: 'error', text: error.message})
            _this.$store.commit('load', false);
        });
      },
    },
    computed: {
      hint: function(){
        return (Math.min(this.ratio[0], this.ratio[1])).toString() + ':' + (Math.max(this.ratio[0], this.ratio[1])-Math.min(this.ratio[0], this.ratio[1])).toString() + ':' + (100-Math.max(this.ratio[0], this.ratio[1])).toString();
      },
      hint1: function(){
        return (this.ratio1.toString() + ':' + (100-this.ratio1).toString());
      }
    },
    mounted(){
      var _this = this;
      for(var i=0; i<this.$store.state.num_classes; i++){
        this.items.push({"text" : this.$store.state.class_labels[i], "value" : "Class " + i})
      }
      setOptions({
        maxParallelUploads: 100,
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
      axios.get(_this.$store.state.server + '/cleartemp');
    }
  }
</script>
