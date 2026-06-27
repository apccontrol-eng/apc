export default {
  title: "Project Docs",
  description: "My VitePress Documentation",

  base: "/apc/",

  themeConfig: {
    nav: [
      { text: "Intro", link: "/intro" },
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
