import React from "react";
import "./Css/App.css";
import { BrowserRouter as Router, Route, Link, Routes } from "react-router-dom";
import Login from "./Login";
// import Account from "./Page/Account";
import Home from "./Home";
import Account from "./Account";
import Diary from "./Diary";

function App() {
  const mode = ["family", "own"];
  var selectedMode = "";
  const mode_select = (target) => {
    selectedMode = target.target.id;
    console.log(selectedMode);
  };
  return (
    <Router>
      <div>
        <header>
          <div className="wrapper">
            <Link to="/Login">
              <button onClick={mode_select} id={mode[0]}>
                Family Diary
              </button>
            </Link>
            <Link to="/Login">
              <button onClick={mode_select} id={mode[1]}>
                Own Diary
              </button>
            </Link>
            <Link to="/Diary">
              <button>Diary</button>
            </Link>
          </div>
        </header>
        <hr />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/Login" element={<Login />} />
          <Route path="/Account" element={<Account />} />
          <Route path="/Diary" element={<Diary />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
