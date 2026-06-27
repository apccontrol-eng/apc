export default {
  title: "Project Docs",
  description: "Intro, MPC, and Process Monitoring",

  themeConfig: {
    nav: [
      { text: "Intro", link: "/intro" },
      { text: "MPC", link: "/mpc" },
      { text: "Process Monitoring", link: "/process-monitoring" },
    ],

    sidebar: {
      "/": [
        {
          text: "Documentation",
          items: [
            { text: "Intro", link: "/intro" },
            { text: "MPC", link: "/mpc" },
            { text: "Process Monitoring", link: "/process-monitoring" },
          ],
        },
      ],
    },
  },
}
