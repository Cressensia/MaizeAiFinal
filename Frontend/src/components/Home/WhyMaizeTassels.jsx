import React from "react";
import "./WhyMaizeTassels.css";
import NavbarLanding from "./NavbarLanding";
import maizetasselFarmer from "./../../images/maizetasselFarmer.png";
import handHoldingCorn from "./../../images/handHoldingCorn.png";
import maizeSunny from "./../../images/maizeSunny.png";
import maizeTasselfield from "./../../images/maizeTasselfield.webp";

export default function WhyMaizeTassels() {
  return (
    <div>
      <NavbarLanding />
      <div className="main-container">
        <h1>Why Maize Tassels?</h1>
        <p className="subtitle">A Plan To Sustainable Agriculture</p>
        <div className="content-block">
          <img src={maizetasselFarmer} />
          <div>
            <h2>A Symbol Of Life And Nourishment</h2>
            <p>
              Maize, one of the world's most vital food resources, stands at the
              forefront of our food security. Its tassels, often overlooked, are
              key to understanding and improving this crucial crop. By studying
              maize tassels, we unlock secrets of genetic diversity, disease
              resistance, and yield potentials.
            </p>
          </div>
        </div>
        <div className="content-block">
          <div>
            <h2>A Symbol Of Life And Nourishment</h2>
            <p>
              Maize, one of the world's most vital food resources, stands at the
              forefront of our food security. Its tassels, often overlooked, are
              key to understanding and improving this crucial crop. By studying
              maize tassels, we unlock secrets of genetic diversity, disease
              resistance, and yield potentials.
            </p>
          </div>
          <img src={maizeSunny} />
        </div>
        <div className="content-block">
          <img src={handHoldingCorn} />
          <div>
            <h2>Innovating for Efficiency</h2>
            <p>
              Our project zeroes in on maize tassels to revolutionize how
              farmers work. We aim to reduce labor intensity and optimize
              agricultural practices. By understanding tassel biology, we
              develop smarter, more efficient farming techniques. This not only
              eases the workload of farmers but also paves the way for higher,
              healthier crop yields.
            </p>
          </div>
        </div>
        <div className="content-block">
          <div>
            <h2>Securing the Future</h2>
            <p>
              In a world facing climatic challenges, our work with maize tassels
              is more than just scientific exploration; it's a step towards
              securing food resources for future generations. By enhancing the
              resilience and productivity of maize, we support global food
              security and the well-being of farming communities worldwide.
            </p>
          </div>
          <img src={maizeTasselfield} />
        </div>
      </div>
    </div>
  );
}
