module.exports = {
  publicPath: process.env.NODE_ENV === 'production'
    ? '/static/'
    : '/',
  chainWebpack: config => {
    config.module.rules.delete('eslint');
  },
  devServer: {
    disableHostCheck: true,
  },

  transpileDependencies: ['vuetify'],

  pluginOptions: {
    i18n: {
      locale: 'en',
      fallbackLocale: 'en',
      localeDir: 'locales',
      enableInSFC: false,
    },
  },
}
