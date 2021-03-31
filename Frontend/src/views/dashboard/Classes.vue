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
    View labels of currently added classes, register labels for new classes
  </v-alert>
  <br/>
  <div>
    <center>
    <h3>Current Classes: </h3><br/>
    <v-col col="12" md="4">
      <v-simple-table fixed-header height="350">
        <template v-slot:default>
          <thead>
            <tr>
              <th class="text-left">
                Class No.
              </th>
              <th class="text-right">
                Class Label
              </th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(i, idx) in $store.state.class_labels">
              <td>{{idx}}</td>
              <td class="text-right"><strong>{{i}}</strong></td>
            </tr>
          </tbody>
        </template>
      </v-simple-table>
    </v-col>
    </center>
    <center>
    <v-col col="12" md="4">
      <v-text-field
        label="Label of new class to add"
        v-model="label"
      ></v-text-field>
    </v-col>
    </center>
    <center>
      <v-btn
        color="primary"
        @click="addClass()"
      >
        Add Class
      </v-btn>
      <!--<v-btn
        color="primary"
        @click="removeClass()"
      >
        Remove Last Added Class
      </v-btn>-->
    </center>
  </div>
  </v-container>
</template>

<style>

</style>

<script>
import axios from 'axios';

  export default {
    name: 'DashboardDashboard',
    components: {

    },
    data () {
      return {
        label: null
      }
    },
    methods: {
      addClass(){
        var _this = this;
        if(this.label){
          axios.post(_this.$store.state.server + '/addClass', {
            class: _this.$store.state.num_classes
          }).then(function (response){
            _this.$store.commit('addClass', _this.label);
            _this.$notify({title: 'Successful', type: 'success', text: response.data})
            _this.$router.go();
          }).catch(function (error){
            _this.$notify({title: 'Error', type: 'error', text: error.message})
          });
        }else{
          _this.$notify({title: 'Error', type: 'error', text: "Class label cannot be blank!"})
        }
      },
      removeClass(){
        var _this = this;
        if(_this.$store.state.num_classes == 43){
          _this.$notify({title: 'Error', type: 'error', text: "Cannot remove an original class!"})
        }else{
          axios.post(_this.$store.state.server + '/removeClass', {
            class: _this.$store.state.num_classes - 1
          }).then(function (response){
            _this.$store.commit('removeClass', true);
            _this.$notify({title: 'Successful', type: 'success', text: response.data})
            _this.$router.go();
          }).catch(function (error){
            _this.$notify({title: 'Error', type: 'error', text: error.message})
          });
        }
      }
    },
    mounted(){

    }
  }
</script>
