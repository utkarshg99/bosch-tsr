import Vue from 'vue'
import Router from 'vue-router'

Vue.use(Router)

export default new Router({
  mode: 'hash',
  base: process.env.BASE_URL,
  routes: [
    {
      path: '/',
      component: () => import('@/views/dashboard/Index'),
      children: [
        {
          name: 'Add Images to Dataset',
          path: 'add',
          component: () => import('@/views/dashboard/Dashboard'),
        },
        {
          name: 'Edit',
          path: 'edit',
          component: () => import('@/views/dashboard/Edit'),
        },
        {
          name: 'Explore Datasets',
          path: '',
          component: () => import('@/views/dashboard/Preview'),
        },
        {
          name: 'Additional Images Added',
          path: 'images',
          component: () => import('@/views/dashboard/Additional'),
        },
        {
          name: 'Make Predictions',
          path: 'custom',
          component: () => import('@/views/dashboard/Custom'),
        },
        {
          name: 'Visualize Classes',
          path: 'detect',
          component: () => import('@/views/dashboard/Ood'),
        },
        {
          name: 'Evaluate Models',
          path: 'evaluate',
          component: () => import('@/views/dashboard/Evaluate'),
        },
        {
          name: 'Improve Models',
          path: 'train',
          component: () => import('@/views/dashboard/Train'),
        },
        {
          name: 'Add New Classes',
          path: 'classes',
          component: () => import('@/views/dashboard/Classes'),
        },
      ],
    },
  ],
})
