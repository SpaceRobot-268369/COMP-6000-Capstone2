import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";

export default function RegisterPage({ accountName, onRegister }) {
  const navigate = useNavigate();
  const [form, setForm] = useState({
    account: accountName,
    email: "",
    password: "",
    confirmPassword: "",
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

    onRegister(nextName);
    navigate("/");
  }

  return (
    <section className="account-page">
      <header className="account-topbar">
        <div>
          <p className="eyebrow">ACCOUNT</p>
          <h1>Register</h1>
          <p className="account-lead">
            Create a simple account to access the platform.
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
              placeholder="Choose an account name"
            />
          </label>

          <label className="field-group">
            <span>Email</span>
            <input
              name="email"
              type="email"
              value={form.email}
              onChange={handleChange}
              placeholder="Enter your email"
            />
          </label>

          <label className="field-group">
            <span>Password</span>
            <input
              name="password"
              type="password"
              value={form.password}
              onChange={handleChange}
              placeholder="Create a password"
            />
          </label>

          <label className="field-group">
            <span>Confirm password</span>
            <input
              name="confirmPassword"
              type="password"
              value={form.confirmPassword}
              onChange={handleChange}
              placeholder="Confirm your password"
            />
          </label>

          <button type="submit" className="auth-primary-btn">
            Register
          </button>

          <p className="account-switch">
            Already have an account? <Link to="/login">Login</Link>
          </p>
        </form>
      </section>
    </section>
  );
}
