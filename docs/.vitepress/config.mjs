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
          { text: "Finite-horizon MPC", link: "/fh-mpc" },
          { text: "Infinite-horizon MPC", link: "/ih-mpc" },
          { text: "PCA", link: "/PCA" },
          { text: "PLS", link: "/PLS" }
        ]
      }
    ]
  }
}
