import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";

export default function LoginPage({ accountName, onLogin }) {
  const navigate = useNavigate();
  const [form, setForm] = useState({
    account: accountName,
    password: "",
  });

  function handleChange(event) {
    const { name, value } = event.target;
    setForm((current) => ({ ...current, [name]: value }));
  }

  function handleSubmit(event) {
    event.preventDefault();
    const nextName = form.account.trim();
    if (!nextName) {
      return;
    }

    onLogin(nextName);
    navigate("/");
  }

  return (
    <section className="account-page">
      <header className="account-topbar">
        <div>
          <p className="eyebrow">ACCOUNT</p>
          <h1>Login</h1>
          <p className="account-lead">
            Sign in with your account name to enter the platform.
          </p>
        </div>
      </header>

      <section className="panel account-card">
        <form className="account-form" onSubmit={handleSubmit}>
          <label className="field-group">
            <span>Account name</span>
            <input
              name="account"
              type="text"
              value={form.account}
              onChange={handleChange}
              placeholder="Enter your account name"
            />
          </label>

          <label className="field-group">
            <span>Password</span>
            <input
              name="password"
              type="password"
              value={form.password}
              onChange={handleChange}
              placeholder="Enter your password"
            />
          </label>

          <button type="submit" className="auth-primary-btn">
            Login
          </button>

          <p className="account-switch">
            No account yet? <Link to="/register">Register</Link>
          </p>
        </form>
      </section>
    </section>
  );
}
