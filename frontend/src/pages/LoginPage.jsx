import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { loginAccount } from "../lib/auth.js";

export default function LoginPage({ accountName, onLogin }) {
  const navigate = useNavigate();
  const [form, setForm] = useState({
    account: accountName,
    password: "",
  });
  const [errorMessage, setErrorMessage] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  function handleChange(event) {
    const { name, value } = event.target;
    setForm((current) => ({ ...current, [name]: value }));
    setErrorMessage("");
  }

  async function handleSubmit(event) {
    event.preventDefault();
    if (!form.account.trim() || !form.password) {
      setErrorMessage("Please enter both your account name and password.");
      return;
    }

    try {
      setIsSubmitting(true);
      const data = await loginAccount({
        account: form.account,
        password: form.password,
      });

      onLogin(data.user.username);
      navigate("/");
    } catch (error) {
      setErrorMessage(error.message);
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <section className="account-page">
      <header className="account-topbar">
        <div>
          <p className="eyebrow">ACCOUNT</p>
          <h1>Login</h1>
          <p className="account-lead">
            Sign in with your account name to access the platform.
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
              autoComplete="username"
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
              autoComplete="current-password"
            />
          </label>

          <button type="submit" className="auth-primary-btn" disabled={isSubmitting}>
            {isSubmitting ? "Logging in..." : "Login"}
          </button>

          {errorMessage ? <p className="account-feedback error">{errorMessage}</p> : null}

          <p className="account-switch">
            No account yet? <Link to="/register">Register</Link>
          </p>
        </form>
      </section>
    </section>
  );
}
