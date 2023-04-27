import { Runtime, Inspector } from "@observablehq/runtime";
import notebook from "./b1fbb11dafedb0d3@96.js";

new Runtime().module(notebook, Inspector.into(document.querySelector("#embedded-notebook")));
