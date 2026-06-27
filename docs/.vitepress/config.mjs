import mathjax3 from 'markdown-it-mathjax3'

export default {
  title: "Project Docs",
  description: "Documentation",

  base: "/apc/",

  markdown: {
    config(md) {
      md.use(mathjax3)
    },
  },

  themeConfig: {
    nav: [
      { text: "MPC", link: "/mpc" },
      { text: "Process Monitoring", link: "/process-monitoring" },
    ],

    sidebar: [
      {
        text: "Docs",
        items: [
          { text: "Intro", link: "/intro" },
          { text: "MPC", link: "/mpc" },
          { text: "Process Monitoring", link: "/process-monitoring" },
        ],
      },
    ],
  },
}
