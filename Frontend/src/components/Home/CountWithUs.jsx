import React, { useState } from "react";
import "./CountWithUs.css";
import NavbarLanding from "./NavbarLanding";
import "../../App.css";

const pricingOptions = {
  monthly: [
    {
      plan: "Free",
      price: "$0",
      period: "lifetime",
      description:
        "Enjoy our $0 plan, always free and designed to provide you with essential features. It's our commitment to accessibility and value.",
      features: [
        "25GB cloud storage",
        "10 visualization plots",
        "Counting services",
        "Phenotype analyser",
        "Disease detection",
      ],
    },
    {
      plan: "Premium",
      price: "$88",
      period: "per month",
      description:
        "Get full feature access, enhanced support, and continuous updates. Ideal for professionals seeking advanced capabilities.",
      features: [
        "50GB cloud storage",
        "10 visualization plots",
        "Counting services",
        "Phenotype analyser",
        "Disease detection",
      ],
    },
    {
      plan: "Enterprise",
      price: "$288",
      period: "per month",
      description:
        "Offers all Premium benefits plus customization, dedicated support, and top-tier security. Perfect for businesses needing a scalable solution.",
      features: [
        "200GB cloud storage",
        "100 visualization plots",
        "Counting services",
        "Phenotype analyser",
        "Disease detection",
      ],
    },
  ],
  yearly: [
    {
      plan: "Free",
      price: "$0",
      period: "lifetime",
      description:
        "Enjoy our $0 plan, always free and designed to provide you with essential features. It's our commitment to accessibility and value.",
      features: [
        "25GB cloud storage",
        "10 visualization plots",
        "Counting services",
        "Phenotype analyser",
        "Disease detection",
      ],
    },
    {
      plan: "Premium",
      price: "$888",
      period: "per year",
      discount: "Save 15%",
      description:
        "Get full feature access, enhanced support, and continuous updates. Ideal for professionals seeking advanced capabilities.",
      features: [
        "50GB cloud storage",
        "10 visualization plots",
        "Counting services",
        "Phenotype analyser",
        "Disease detection",
      ],
    },
    {
      plan: "Enterprise",
      price: "$3088",
      period: "per year",
      discount: "Save 10%",
      description:
        "Offers all Premium benefits plus customization, dedicated support, and top-tier security. Perfect for businesses needing a scalable solution.",
      features: [
        "200GB cloud storage",
        "100 visualization plots",
        "Counting services",
        "Phenotype analyser",
        "Disease detection",
      ],
    },
  ],
};

export default function CountWithUs() {
  const [selectedOption, setSelectedOption] = useState("monthly");

  return (
    <div>
      <NavbarLanding />
      <div className="main-container">
        <div className="gradient-background">
          <div className="pricing-header">
            <h2>Tailored Pricing Solutions</h2>
            <p>
              Our Services Are Designed To Cater Your Specific Needs And Goals
            </p>
            <div className="toggle-buttons">
              <button
                onClick={() => setSelectedOption("monthly")}
                className={selectedOption === "monthly" ? "active" : ""}
              >
                Monthly
              </button>
              <button
                onClick={() => setSelectedOption("yearly")}
                className={selectedOption === "yearly" ? "active" : ""}
              >
                Annually
              </button>
            </div>
          </div>
          <div className="pricing-cards">
            {pricingOptions[selectedOption].map((option, index) => (
              <div key={index} className="pricing-card">
                <h3>{option.plan}</h3>
                <p className="price">{option.price}</p>
                <span>{option.period}</span>
                <p className="description">{option.description}</p>
                {option.discount && (
                  <p className="discount">{option.discount}</p>
                )}
                <ul>
                  {option.features.map((feature, index) => (
                    <li key={index}>{feature}</li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
