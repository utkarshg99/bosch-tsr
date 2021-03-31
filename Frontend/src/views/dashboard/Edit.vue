<template>
  <v-container
    id="dashboard"
    fluid
    tag="section"
  >
  <center>
  <router-link to="/images">
    <v-btn
      color="primary"
      text
    >Cancel</v-btn>
  </router-link>
  <v-btn
    color="primary"
    text
    @click="image = $refs.tuiImageEditor.invoke('toDataURL'); save()"
  >Save</v-btn>
  <v-btn
    color="primary"
    text
    @click="image = $refs.tuiImageEditor.invoke('toDataURL'); saveCopy()"
  >Save a copy</v-btn>
  </center>
  <br/>
        <div class="imageEditorApp">
          <tui-image-editor :include-ui="useDefaultUI" :options="options" ref="tuiImageEditor"></tui-image-editor>
        </div>
  </v-container>
</template>

<style>
.imageEditorApp {
    width: 100%;
    height: 600px;
}
.tui-image-editor-header-buttons {
   visibility: hidden !important;
}
</style>

<script>
import axios from 'axios';

import {ImageEditor} from '@toast-ui/vue-image-editor';
import 'tui-image-editor/dist/tui-image-editor.css';

  export default {
    name: 'DashboardDashboard',
    components: {
      'tui-image-editor': ImageEditor
    },
    data () {
      return {
        image: null,
        useDefaultUI: true,
        options: { // for tui-image-editor component's "options" prop
          includeUI: {
                    loadImage: {
                        path: null,
                        name: null
                    },
                    initMenu: '',
                    menuBarPosition: 'bottom',
                    menu: ['crop', 'flip', 'rotate', 'filter'],
                },
                usageStatistics: false,
                cssMaxWidth: 700,
                cssMaxHeight: 500
        },
      }
    },
    methods: {
      complete (index) {
        this.list[index] = !this.list[index]
      },
      save(){
        var _this = this;
        axios.post(_this.$store.state.server + '/save', {
            path: _this.$store.state.sel_image,
            type: _this.$store.state.sel_type,
            image: _this.image
        }).then(function (response){
            _this.$notify({title: 'Successful', type: 'success', text: "Successfully saved"})
            if(response.data=="Successful"){
              _this.$router.push("images")
            }
        }).catch(function (error){
            _this.$notify({title: 'Error', type: 'error', text: error.message})
        });
      },
      saveCopy(){
        var _this = this;
        axios.post(_this.$store.state.server + '/saveCopy', {
            path: _this.$store.state.sel_image,
            type: _this.$store.state.sel_type,
            image: _this.image
        }).then(function (response){
            _this.$notify({title: 'Successful', type: 'success', text: "Successfully saved a copy"})
            if(response.data=="Successful"){
              _this.$router.push("images")
            }
        }).catch(function (error){
            _this.$notify({title: 'Error', type: 'error', text: error.message})
        });
      }
    },
    computed: {
      date: function(){
        var v = new Date()
        return v.getTime();
      }
    },
    mounted () {
      let actions = this.$refs.tuiImageEditor.invoke('getActions');
      if(this.$store.state.sel_image && actions) {
        actions.main.initLoadImage(this.$store.state.sel_image + '?ver=' + this.date, 'My sample image');
        this.$refs.tuiImageEditor.invoke('ui.activeMenuEvent');
      }
    }
  }
</script>
