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
    nav: [
      { text: "Home", link: "/" }
    ],

    sidebar: [
      {
        text: "Documentation",
        collapsed: false, // true = starts collapsed
        items: [
          { text: "MPC", link: "/mpc" },
          { text: "Process Monitoring", link: "/process-monitoring" },
          { text: "Getting Started", link: "/getting-started" }
        ]
      }
    ]
  }
}
