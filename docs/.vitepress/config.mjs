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
            text: "Infinite Horizon",
            link: "/mpc/ih-mpc"
          },
          {
            text: "Finite Horizon",
            link: "/mpc/fh-mpc"
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
