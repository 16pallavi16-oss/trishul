"use client";

import { useState, useEffect } from "react";

export default function Home() {
  const [reply, setReply] = useState("Loading...");

  useEffect(() => {
    fetch("http://localhost:8000/hello")
      .then((res) => res.json())
      .then((data) => {
        console.log("Got data:", data);
        setReply(data.ollama); // <-- changed from data.reply to data.ollama
      })
      .catch((err) => console.log("Fetch failed:", err));
}, []);

  return (
    <div>
      <h1>Hello World</h1>
      <p>{reply}</p>
    </div>
  );
}