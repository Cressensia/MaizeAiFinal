import React from 'react';
import './GridsForFTV.css'; 

const getCircleColor = (count) => {
  if (count > 300) return 'green';
  if (count > 150) return 'orange';
  return 'red';
};

const GridsForFTV = ({ plots }) => {
  return (
    <div className="grids-for-ftv">
      {plots.map(plot => (
        <div className="plot-grid" key={plot.plotName}>
          <h3 className="plot-title">{plot.plotName}</h3>
          <div className="sections-grid">
            {plot.sections.map((section) => (
              <div className="section" key={`${plot.plotName}-section-${section.section}`}>
                <div className={`circle ${getCircleColor(section.tasselCount)}`}>
                  {section.tasselCount}
                </div>
                <div className="section-label">Section {section.section}</div>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
};

export default GridsForFTV;