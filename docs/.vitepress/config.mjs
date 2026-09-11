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
      },
      {
        text: "DMC",
        collapsed: false,
        items: [
          {
            text: "Dynamic Matrix Control",
            link: "/dmc/dmc"
          },
          {
            text: "Example",
            link: "/dmc/dmc-example"
          }
        ]
      },
      {
        text: "State Estimation",
        collapsed: false,
        items: [
          {
            text: "Kalman Filter",
            link: "/state-estimation/KF"
          }
        ]
      },
      {
        text: "Subspace Identification",
        collapsed: false
      },
      {
        text: "Quadratic Programming",
        collapsed: false
      },
      {
        text: "Miscellaneous",
        collapsed: false,
        items: [
          {
            text: "Aurafarmingjpg",
            link: "/misc/misc"
          }
        ]
      }
    ]
  }
}
