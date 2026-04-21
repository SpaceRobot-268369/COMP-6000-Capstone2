import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { registerAccount } from "../lib/auth.js";

export default function RegisterPage({ accountName, onRegister }) {
  const navigate = useNavigate();
  const [form, setForm] = useState({
    account: accountName,
    email: "",
    password: "",
    confirmPassword: "",
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
    const nextName = form.account.trim();
    if (!nextName || !form.email.trim() || !form.password) {
      setErrorMessage("Please complete all required fields.");
      return;
    }

    if (form.password !== form.confirmPassword) {
      setErrorMessage("The passwords do not match.");
      return;
    }

    try {
      setIsSubmitting(true);
      const data = await registerAccount({
        username: form.account,
        email: form.email,
        password: form.password,
      });

      onRegister(data.user.username);
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
          <h1>Register</h1>
          <p className="account-lead">
            Create an account to access the platform.
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
              autoComplete="username"
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
              autoComplete="email"
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
              autoComplete="new-password"
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
              autoComplete="new-password"
            />
          </label>

          <button type="submit" className="auth-primary-btn" disabled={isSubmitting}>
            {isSubmitting ? "Registering..." : "Register"}
          </button>

          {errorMessage ? <p className="account-feedback error">{errorMessage}</p> : null}

          <p className="account-switch">
            Already have an account? <Link to="/login">Login</Link>
          </p>
        </form>
      </section>
    </section>
  );
}
