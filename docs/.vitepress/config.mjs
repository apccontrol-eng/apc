import mathjax3 from 'markdown-it-mathjax3'

export default {
  title: "Project Docs",
  base: "/apc/",

  markdown: {
    config(md) {
      md.use(mathjax3)
    },
  },

  themeConfig: {
    sidebar: [
      {
        text: "Docs",
        items: [
          { text: "MPC", link: "/mpc" },
          { text: "Process Monitoring", link: "/process-monitoring" }
        ]
      }
    ]
  }
}
