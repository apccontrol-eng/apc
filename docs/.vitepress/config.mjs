import mathjax3 from 'markdown-it-mathjax3'

export default {
  title: "Project Docs",
  base: "/apc/",

  markdown: {
    config(md) {
      md.use(mathjax3)
    }
  },
  
  themeConfig: {
    sidebar: [
      {
        text: "MPC",
        collapsed: false,
        items: [
          {
            text: "Finite-horizon",
            link: "/mpc/fh-mpc"
          },
          {
            text: "Infinite-horizon",
            link: "/mpc/ih-mpc"
          }
        ]
      },
      {
        text: "Process Monitoring",
        collapsed: false,
        items: [
          {
            text: "PCA",
            link: "/process-monitoring/PCA"
          },
          {
            text: "PLS",
            link: "/process-monitoring/PLS"
          }
        ]
      }
    ]
  }
}
