<template>
  <div>
  <v-overlay :value="$store.state.loading" style="z-index:1000">
    <v-progress-circular
      indeterminate
      size="80"
    ></v-progress-circular>
  </v-overlay>
  <notifications position="top center"/>
  <router-view />
  </div>
</template>

<style>
.vue-notification {
  font-size: 14px !important;
  font-family: Arial !important;
}
</style>

<script>
  export default {
    name: 'App',
  }
</script>
