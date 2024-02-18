import React from "react";
import { Link } from "react-router-dom";

function Login(props) {
  return (
    <div>
      <h1>Login</h1>
      <p>Welcom login page!</p>
      <Link to="/Account">
        <button>Account</button>
      </Link>
    </div>
  );
}

export default Login;
