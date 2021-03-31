import Vue from 'vue'
import Vuex from 'vuex'
import createPersistedState from 'vuex-persistedstate'

Vue.use(Vuex)

export default new Vuex.Store({
  plugins: [createPersistedState({
    storage: window.localStorage,
    reducer: state => ({
      class_labels: state.class_labels,
      num_classes: state.num_classes
    })
  }), createPersistedState({
    storage: window.sessionStorage,
    reducer: state => ({
      sel_image: state.sel_image,
      sel_type: state.sel_type,
      sel: state.sel,
      selectedTab: state.selectedTab
    })
  })],
  state: {
    tsne_plot: null,
    show_plot: false,
    eval_prog: [0, 0, 0, 0, 0],
    server: "https://boschtsr.ml",
    sel_image: null,
    sel_type: null,
    sel: "Training Dataset",
    selectedTab: 0,
    loading: false,
    barColor: 'rgba(0, 0, 0, .8), rgba(0, 0, 0, .8)',
    barImage: 'https://i0.wp.com/blogs.cfainstitute.org/investor/files/2018/01/Artificial-Intelligence-Machine-Learning-and-Deep-Learning-A-Primer.png?resize=940%2C575&ssl=1',
    drawer: null,
    num_classes: 48,
    class_labels: [
      "Speed limit (20km/h)",
      "Speed limit (30km/h)",
      "Speed limit (50km/h)",
      "Speed limit (60km/h)",
      "Speed limit (70km/h)",
      "Speed limit (80km/h)",
      "End of speed limit (80km/h)",
      "Speed limit (100km/h)",
      "Speed limit (120km/h)",
      "No passing",
      "No passing vehicle over 3.5 tons",
      "Right-of-way at intersection",
      "Priority road",
      "Yield",
      "Stop",
      "No vehicles",
      "Vehicles > 3.5 tons prohibited",
      "No entry",
      "General caution",
      "Dangerous curve left",
      "Dangerous curve right",
      "Double curve",
      "Bumpy road",
      "Slippery road",
      "Road narrows on the right",
      "Road work",
      "Traffic signals",
      "Pedestrians",
      "Children crossing",
      "Bicycles crossing",
      "Beware of ice/snow",
      "Wild animals crossing",
      "End speed + passing limits",
      "Turn right ahead",
      "Turn left ahead",
      "Ahead only",
      "Go straight or right",
      "Go straight or left",
      "Keep right",
      "Keep left",
      "Roundabout mandatory",
      "End of no passing",
      "End no passing vehicle > 3.5 tons",
      "Maximum Speed Limit 40",
      "Limited access road",
      "Side road junction on the right",
      "No stopping",
      "No honking"
    ],
  },
  mutations: {
    SET_BAR_IMAGE (state, payload) {
      state.barImage = payload
    },
    SET_DRAWER (state, payload) {
      state.drawer = payload
    },
    load (state, payload) {
      state.loading = payload
    },
    selImage (state, payload) {
      state.sel_image = payload
    },
    selType (state, payload) {
      state.sel_type = payload
    },
    addClass (state, payload) {
      state.class_labels.push(payload);
      state.num_classes++;
    },
    removeClass (state, payload) {
      state.class_labels.pop();
      state.num_classes--;
    },
    updateSel (state, payload) {
      state.sel = payload
    },
    updateTab (state, payload) {
      state.selectedTab = payload
    },
    updateEvalProg (state, payload) {
      state.eval_prog = payload
    },
    settsne (state, payload) {
      state.tsne_plot = payload
    },
    setplot (state, payload) {
      state.show_plot = payload
    }
  },
  actions: {

  },
})
