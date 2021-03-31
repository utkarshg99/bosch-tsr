import Vue from 'vue'
import App from './App.vue'
import router from './router'
import store from './store'
import './plugins/base'
import './plugins/apexcharts'
import './plugins/vee-validate'
import vuetify from './plugins/vuetify'
import i18n from './i18n'
import Notifications from 'vue-notification'

Vue.config.productionTip = false
Vue.use(Notifications)

new Vue({
  router,
  store,
  vuetify,
  i18n,
  render: h => h(App),
}).$mount('#app')
